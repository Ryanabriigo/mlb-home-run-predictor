
import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import date, timedelta
import math
import time
from pybaseball import statcast_batter, statcast_pitcher

# ============================================================
# MLB DAILY HOME RUN PREDICTOR — v4
# ============================================================

API = "https://statsapi.mlb.com/api/v1"
LEAGUE_HR_PA = 0.035
DEFAULT_BARREL = 0.070
DEFAULT_HARD_HIT = 0.370
DEFAULT_EV = 88.5

st.set_page_config(
    page_title="MLB Daily Home Run Predictor",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -------------------------
# Styling
# -------------------------
st.markdown(
    """
    <style>
    .pick-card {
        padding: 18px;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,.25);
        margin-bottom: 12px;
    }
    .big-number { font-size: 2rem; font-weight: 700; }
    @media (max-width: 700px) {
        .pick-card { padding: 14px; border-radius: 12px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# MLB API helpers
# -------------------------
@st.cache_data(ttl=300)
def api_get(path, params=None):
    r = requests.get(f"{API}/{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=300)
def get_games(game_date):
    data = api_get(
        "schedule",
        {
            "sportId": 1,
            "date": game_date,
            "hydrate": "probablePitcher,team,venue",
        },
    )

    rows = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            away = g["teams"]["away"]
            home = g["teams"]["home"]

            rows.append(
                {
                    "game_pk": g["gamePk"],
                    "game_date": game_date,
                    "away": away["team"].get("abbreviation", away["team"]["name"]),
                    "home": home["team"].get("abbreviation", home["team"]["name"]),
                    "away_id": away["team"]["id"],
                    "home_id": home["team"]["id"],
                    "away_sp": away.get("probablePitcher", {}),
                    "home_sp": home.get("probablePitcher", {}),
                    "venue": g.get("venue", {}).get("name", "Unknown"),
                    "venue_lat": g.get("venue", {}).get("location", {}).get("latitude"),
                    "venue_lon": g.get("venue", {}).get("location", {}).get("longitude"),
                    "game_time": g.get("gameDate"),
                    "status": g.get("status", {}).get("detailedState", ""),
                }
            )
    return pd.DataFrame(rows)


@st.cache_data(ttl=1800)
def get_team_hitters(team_id, season):
    data = api_get(
        f"teams/{team_id}/roster",
        {"season": season, "hydrate": "person"},
    )

    rows = []
    for item in data.get("roster", []):
        person = item.get("person", {})
        pos = item.get("position", {}).get("abbreviation", "")
        if pos not in {"P", "TWP"}:
            rows.append(
                {
                    "id": person.get("id"),
                    "name": person.get("fullName"),
                    "position": pos,
                }
            )
    return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def get_lineup(game_pk, side):
    """Return confirmed batting order from the MLB live game feed.

    A batting order is considered confirmed only when MLB exposes
    battingOrder values in the live feed.
    """
    try:
        data = api_get(f"game/{game_pk}/feed/live")
        team = (
            data.get("liveData", {})
            .get("boxscore", {})
            .get("teams", {})
            .get(side, {})
        )
        players = team.get("players", {})

        lineup = []
        for p in players.values():
            bo = p.get("battingOrder")
            if bo:
                try:
                    order = int(bo) // 100
                except Exception:
                    continue

                if 1 <= order <= 9:
                    lineup.append(
                        {
                            "id": p["person"]["id"],
                            "name": p["person"]["fullName"],
                            "order": order,
                        }
                    )

        if not lineup:
            return pd.DataFrame(columns=["id", "name", "order"])

        return pd.DataFrame(lineup).sort_values("order").drop_duplicates("id")
    except Exception:
        return pd.DataFrame(columns=["id", "name", "order"])


# -------------------------
# Statcast helpers — v10.1 safe speed layer
# -------------------------
@st.cache_data(ttl=21600, show_spinner=False)
def get_batter_statcast_raw(player_id, start_date, end_date):
    """Download one hitter's long window once and reuse it locally.

    The old app downloaded separate 15-day, 90-day and BvP datasets for
    every hitter. This function makes one 90-day request per hitter; the
    recent-window stats and BvP stats are derived locally from that same
    dataframe.
    """
    if not player_id:
        return pd.DataFrame()
    try:
        df = statcast_batter(start_date, end_date, player_id)
        if df is None:
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()


def _empty_batter_summary():
    return {
        "pa": 0,
        "hr": 0,
        "hr_pa": LEAGUE_HR_PA,
        "barrel_rate": DEFAULT_BARREL,
        "hard_hit_rate": DEFAULT_HARD_HIT,
        "ev": DEFAULT_EV,
        "bbe": 0,
    }


def summarize_batter_df(df):
    if df is None or df.empty:
        return _empty_batter_summary()

    pa_df = df[df["events"].notna()].copy() if "events" in df.columns else pd.DataFrame()
    bbe = df[df["launch_speed"].notna()].copy() if "launch_speed" in df.columns else pd.DataFrame()

    pa = len(pa_df)
    hr = int((pa_df["events"] == "home_run").sum()) if pa else 0

    if len(bbe):
        hard_hit = float((bbe["launch_speed"] >= 95).mean())
        ev = float(bbe["launch_speed"].mean())
    else:
        hard_hit = DEFAULT_HARD_HIT
        ev = DEFAULT_EV

    if len(bbe) and "launch_speed_angle" in bbe.columns:
        barrel_rate = float((bbe["launch_speed_angle"] == 6).mean())
    else:
        barrel_rate = DEFAULT_BARREL

    reliability = min(1.0, pa / 120.0)
    hr_pa = reliability * (hr / max(pa, 1)) + (1 - reliability) * LEAGUE_HR_PA
    barrel_rate = reliability * barrel_rate + (1 - reliability) * DEFAULT_BARREL
    hard_hit = reliability * hard_hit + (1 - reliability) * DEFAULT_HARD_HIT
    ev = reliability * ev + (1 - reliability) * DEFAULT_EV

    return {
        "pa": pa,
        "hr": hr,
        "hr_pa": float(hr_pa),
        "barrel_rate": float(barrel_rate),
        "hard_hit_rate": float(hard_hit),
        "ev": float(ev),
        "bbe": len(bbe),
    }


def build_batter_profiles(df, recent_start, long_start):
    """Build recent and long-window summaries from one cached 90-day pull."""
    if df is None or df.empty:
        empty = _empty_batter_summary()
        return empty.copy(), empty.copy()

    recent = df
    if "game_date" in df.columns:
        dates = pd.to_datetime(df["game_date"], errors="coerce")
        recent = df[dates >= pd.Timestamp(recent_start)].copy()

    return summarize_batter_df(recent), summarize_batter_df(df)


def summarize_batter_pitcher_matchup(df, pitcher_id):
    """Calculate BvP locally from the hitter's already-loaded Statcast data."""
    if not pitcher_id or df is None or df.empty or "pitcher" not in df.columns:
        return {"pa": 0, "hr": 0, "hr_pa": LEAGUE_HR_PA}

    try:
        matchup_df = df[df["pitcher"].eq(int(pitcher_id))]
        if matchup_df.empty or "events" not in matchup_df.columns:
            return {"pa": 0, "hr": 0, "hr_pa": LEAGUE_HR_PA}

        pa_df = matchup_df[matchup_df["events"].notna()]
        pa = len(pa_df)
        hr = int((pa_df["events"] == "home_run").sum())
        raw = hr / max(pa, 1)
        reliability = min(1.0, pa / 30.0)
        shrunk = reliability * raw + (1.0 - reliability) * LEAGUE_HR_PA
        return {"pa": pa, "hr": hr, "hr_pa": float(shrunk)}
    except Exception:
        return {"pa": 0, "hr": 0, "hr_pa": LEAGUE_HR_PA}


@st.cache_data(ttl=21600, show_spinner=False)
def get_pitcher_window(player_id, start_date, end_date):
    if not player_id:
        return {"hr_rate": LEAGUE_HR_PA, "hr": 0, "pa": 0}

    try:
        df = statcast_pitcher(start_date, end_date, player_id)
    except Exception:
        return {"hr_rate": LEAGUE_HR_PA, "hr": 0, "pa": 0}

    if df is None or df.empty:
        return {"hr_rate": LEAGUE_HR_PA, "hr": 0, "pa": 0}

    pa_df = df[df["events"].notna()].copy()
    pa = len(pa_df)
    hr = int((pa_df["events"] == "home_run").sum())

    raw = hr / max(pa, 1)
    reliability = min(1.0, pa / 180.0)
    shrunk = reliability * raw + (1 - reliability) * LEAGUE_HR_PA

    return {"hr_rate": float(shrunk), "hr": hr, "pa": pa}


@st.cache_data(ttl=900, show_spinner=False)
def get_weather(lat, lon, game_date, game_time=None):
    """Fetch a lightweight game-time weather estimate from Open-Meteo."""
    neutral = {"condition": "Unknown", "temp_f": np.nan, "wind_mph": np.nan}
    if lat is None or lon is None:
        return neutral
    try:
        params = {
            "latitude": float(lat),
            "longitude": float(lon),
            "hourly": "temperature_2m,wind_speed_10m,weather_code",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "timezone": "auto",
            "forecast_days": 16,
        }
        r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        times = pd.to_datetime(data.get("hourly", {}).get("time", []))
        if len(times) == 0:
            return neutral
        target = pd.to_datetime(game_time) if game_time else pd.to_datetime(game_date + "T19:00:00")
        if target.tzinfo is not None:
            target = target.tz_localize(None)
        idx = int(np.argmin(np.abs(times - target)))
        temp = data.get("hourly", {}).get("temperature_2m", [np.nan])[idx]
        wind = data.get("hourly", {}).get("wind_speed_10m", [np.nan])[idx]
        code = data.get("hourly", {}).get("weather_code", [None])[idx]
        labels = {
            0: "Clear", 1: "Mostly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Fog", 48: "Fog", 51: "Drizzle", 53: "Drizzle", 55: "Drizzle",
            61: "Rain", 63: "Rain", 65: "Heavy rain", 71: "Snow", 73: "Snow", 75: "Snow",
            80: "Showers", 81: "Showers", 82: "Heavy showers", 95: "Thunderstorms",
        }
        return {
            "condition": labels.get(int(code), "Unknown") if code is not None else "Unknown",
            "temp_f": float(temp) if pd.notna(temp) else np.nan,
            "wind_mph": float(wind) if pd.notna(wind) else np.nan,
        }
    except Exception:
        return neutral


def weather_multiplier(weather):
    """Keep weather effects intentionally small and conservative."""
    temp = weather.get("temp_f")
    wind = weather.get("wind_mph")
    mult = 1.0
    if pd.notna(temp):
        if temp >= 85:
            mult += 0.025
        elif temp <= 45:
            mult -= 0.025
    if pd.notna(wind) and wind >= 15:
        mult += 0.015
    return clamp(mult, 0.95, 1.05)


# -------------------------
# Model
# -------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def model_player(batter_30, batter_90, pitcher_90, confirmed, lineup_slot, matchup=None, weather=None):
    # Recent form gets meaningful weight, while 90-day performance stabilizes
    # estimates when the recent sample is small.
    hr_signal = (
        0.60 * (batter_30["hr_pa"] / LEAGUE_HR_PA)
        + 0.40 * (batter_90["hr_pa"] / LEAGUE_HR_PA)
    )

    barrel_signal = batter_30["barrel_rate"] / DEFAULT_BARREL
    hard_hit_signal = batter_30["hard_hit_rate"] / DEFAULT_HARD_HIT
    ev_signal = batter_30["ev"] / DEFAULT_EV
    pitcher_signal = pitcher_90["hr_rate"] / LEAGUE_HR_PA
    matchup_signal = (matchup or {}).get("hr_pa", LEAGUE_HR_PA) / LEAGUE_HR_PA
    matchup_signal = clamp(matchup_signal, 0.65, 1.55)
    weather_mult = weather_multiplier(weather or {})

    hitter_quality = (
        0.42 * hr_signal
        + 0.23 * barrel_signal
        + 0.20 * hard_hit_signal
        + 0.10 * ev_signal
        + 0.05 * matchup_signal
    )
    hitter_quality = clamp(hitter_quality, 0.35, 3.5)
    pitcher_quality = clamp(0.65 + 0.35 * pitcher_signal, 0.55, 2.25)

    pa_by_slot = {
        1: 4.55, 2: 4.45, 3: 4.35, 4: 4.25, 5: 4.10,
        6: 3.95, 7: 3.80, 8: 3.65, 9: 3.55
    }
    slot = int(lineup_slot) if pd.notna(lineup_slot) else 5
    expected_pa = pa_by_slot.get(slot, 3.75)

    # Pending lineups should never receive the same opportunity assumption as
    # a confirmed starter. Keep a modest PA assumption but apply an explicit
    # probability/score penalty so non-confirmed players do not dominate the board.
    lineup_penalty = 1.00 if confirmed else 0.82
    if not confirmed:
        expected_pa = min(expected_pa, 3.75)

    adjusted_hr_pa = LEAGUE_HR_PA * hitter_quality * pitcher_quality * weather_mult
    adjusted_hr_pa = clamp(adjusted_hr_pa, 0.005, 0.12)
    probability = 1.0 - (1.0 - adjusted_hr_pa) ** expected_pa
    probability *= lineup_penalty

    sample_score = clamp(
        (batter_30["pa"] / 80.0) * 50 + (batter_90["pa"] / 220.0) * 50,
        0, 100
    )
    lineup_score = 100 if confirmed else 35
    data_confidence = 0.65 * sample_score + 0.35 * lineup_score

    if data_confidence >= 80:
        confidence = "HIGH"
    elif data_confidence >= 60:
        confidence = "GOOD"
    else:
        confidence = "WATCH"

    # Score is intentionally not a second saturated probability model.
    # It is a human-readable 0-100 index tied to the predicted HR probability,
    # which prevents a large cluster of players from all receiving 100.
    raw_score = 100.0 * clamp(probability / 0.45, 0.0, 1.0)
    score = raw_score * (0.92 if not confirmed else 1.0)

    return {
        "probability": float(probability),
        "score": float(score),
        "expected_pa": float(expected_pa),
        "confidence": confidence,
        "data_confidence": float(data_confidence),
        "lineup_penalty": lineup_penalty,
        "weather_multiplier": weather_mult,
        "matchup_pa": (matchup or {}).get("pa", 0),
    }


def reason(row):
    reasons = []
    if row["Barrel %"] >= 10:
        reasons.append("elite barrel rate")
    elif row["Barrel %"] >= 8:
        reasons.append("strong barrel rate")

    if row["Hard-Hit %"] >= 45:
        reasons.append("elite hard contact")
    elif row["Hard-Hit %"] >= 40:
        reasons.append("strong hard contact")

    if row["Avg EV"] >= 91:
        reasons.append("excellent exit velocity")

    if row["Pitcher HR/PA"] >= 0.045:
        reasons.append("favorable pitcher HR tendency")

    if row["Lineup"] == "Confirmed":
        reasons.append(f"confirmed #{int(row['Batting Order'])} lineup spot")

    if not reasons:
        reasons.append("balanced hitter/pitcher matchup")

    return ", ".join(reasons)


def pick_label(row, rank, confirmed_count):
    """Give the daily board useful, human-readable pick types."""
    if rank == 1:
        return "🥇 BEST PICK"
    if rank == 2:
        return "🥈 STRONG PICK"
    if rank == 3:
        return "🥉 STRONG PICK"
    if row["Confidence"] == "HIGH" and row["Score"] >= 70:
        return "💪 STRONG PICK"
    if row["Lineup"] == "Confirmed" and row["Score"] >= 60 and row["HR Probability"] >= 0.10:
        return "💎 VALUE PICK"
    if row["Lineup"] == "Confirmed" and row["Barrel %"] >= 8:
        return "🌙 SLEEPER PICK"
    return "👀 WATCH"



# -------------------------
# Persistent prediction history
# -------------------------
HISTORY_FILE = "prediction_history.csv"

def load_prediction_history():
    try:
        return pd.read_csv(HISTORY_FILE)
    except Exception:
        return pd.DataFrame(columns=[
            "date", "player", "team", "opponent", "probability",
            "score", "confidence", "rank", "actual_hr"
        ])

def save_prediction_history(df):
    if df is not None and not df.empty:
        df.to_csv(HISTORY_FILE, index=False)

def record_predictions(result_df, prediction_date):
    hist = load_prediction_history()
    new = result_df.copy()
    new["date"] = prediction_date
    new["rank"] = np.arange(1, len(new) + 1)
    new["actual_hr"] = np.nan
    keep = ["date", "Player", "Team", "Opponent", "HR Probability",
            "Score", "Confidence", "rank", "actual_hr"]
    new = new[keep].rename(columns={
        "Player": "player", "Team": "team", "Opponent": "opponent",
        "HR Probability": "probability", "Score": "score",
        "Confidence": "confidence"
    })
    # Replace predictions for the same date/player rather than duplicating them.
    hist = hist[~(
        hist["date"].astype(str).eq(str(prediction_date)) &
        hist["player"].isin(new["player"])
    )]
    combined = pd.concat([hist, new], ignore_index=True)
    save_prediction_history(combined)
    return combined

def performance_summary(hist):
    checked = hist.dropna(subset=["actual_hr"]).copy()
    if checked.empty:
        return None
    checked["actual_hr"] = checked["actual_hr"].astype(int)
    top1 = checked[checked["rank"] == 1]["actual_hr"].mean()
    top5 = checked[checked["rank"] <= 5]["actual_hr"].mean()
    top10 = checked[checked["rank"] <= 10]["actual_hr"].mean()
    brier = ((checked["probability"] - checked["actual_hr"]) ** 2).mean()
    return {
        "checked": len(checked),
        "top1": top1 if pd.notna(top1) else 0,
        "top5": top5 if pd.notna(top5) else 0,
        "top10": top10 if pd.notna(top10) else 0,
        "brier": brier,
    }

# -------------------------
# UI
# -------------------------
st.title("⚾ MLB Daily Home Run Predictor")
st.caption(
    "Statcast-based estimate of each hitter's chance to hit at least one home run. "
    "Use HR Probability as the primary signal; Model Score is a normalized 0–100 index."
)

col1, col2, col3 = st.columns(3)
with col1:
    selected_date = st.date_input("Game date", date.today())
with col2:
    recent_days = st.slider("Statcast lookback (days)", 7, 45, 15)
with col3:
    top_n = st.slider("Show top candidates", 5, 30, 30)

run = st.button(
    "🚀 RUN TODAY'S PREDICTIONS",
    type="primary",
    use_container_width=True,
)

if run:
    games_df = get_games(selected_date.isoformat())
    if games_df.empty:
        st.warning("No MLB games were found for this date.")
        st.stop()

    season = selected_date.year
    start_30 = selected_date - timedelta(days=recent_days)
    start_90 = selected_date - timedelta(days=90)
    rows = []
    progress = st.progress(0, text="⚡ Preparing today's slate…")
    total_steps = max(len(games_df) * 2, 1)
    completed = 0
    started_at = time.perf_counter()
    unique_hitter_ids = set()
    unique_pitcher_ids = set()

    for _, game in games_df.iterrows():
        for side in ("away", "home"):
            team = game[side]
            team_id = int(game[f"{side}_id"])
            opponent = game["home"] if side == "away" else game["away"]
            pitcher = game["home_sp"] if side == "away" else game["away_sp"]
            pitcher_id = pitcher.get("id")
            pitcher_name = pitcher.get("fullName", "TBD")

            p90 = get_pitcher_window(pitcher_id, start_90.isoformat(), selected_date.isoformat())
            weather = get_weather(game.get("venue_lat"), game.get("venue_lon"), selected_date.isoformat(), game.get("game_time"))
            lineup = get_lineup(int(game["game_pk"]), side)
            confirmed = not lineup.empty
            hitters = get_team_hitters(team_id, season)

            if confirmed:
                hitters = hitters.merge(lineup[["id", "order"]], on="id", how="inner")
                hitters["order"] = hitters["order"].astype(int)
            else:
                hitters["order"] = np.nan

            # Track how much work this run actually needs.
            unique_hitter_ids.update(int(x) for x in hitters["id"].dropna().tolist())
            if pitcher_id:
                unique_pitcher_ids.add(int(pitcher_id))

            progress.progress(
                completed / total_steps,
                text=f"⚡ Loading {team} data…",
            )

            for _, hitter in hitters.iterrows():
                pid = int(hitter["id"])
                # One cached 90-day pull now powers recent stats, long-term stats,
                # and BvP. This removes two redundant Statcast downloads per hitter.
                batter_raw = get_batter_statcast_raw(
                    pid,
                    start_90.isoformat(),
                    selected_date.isoformat(),
                )
                b30, b90 = build_batter_profiles(
                    batter_raw,
                    start_30.isoformat(),
                    start_90.isoformat(),
                )
                matchup = summarize_batter_pitcher_matchup(batter_raw, pitcher_id)
                slot = hitter["order"] if pd.notna(hitter["order"]) else 5

                result = model_player(
                    b30, b90, p90, confirmed, slot,
                    matchup=matchup, weather=weather,
                )

                rows.append({
                    "Player": hitter["name"],
                    "Team": team,
                    "Opponent": opponent,
                    "Opposing SP": pitcher_name,
                    "Venue": game["venue"],
                    "Game PK": int(game["game_pk"]),
                    "Player ID": pid,
                    "Weather": weather.get("condition", "Unknown"),
                    "Temp F": weather.get("temp_f"),
                    "Wind MPH": weather.get("wind_mph"),
                    "Matchup PA": matchup.get("pa", 0),
                    "Matchup HR/PA": matchup.get("hr_pa", LEAGUE_HR_PA),
                    "HR Probability": result["probability"],
                    "Score": result["score"],
                    "Confidence": result["confidence"],
                    "HR/PA": b30["hr_pa"],
                    "Barrel %": b30["barrel_rate"],
                    "Hard-Hit %": b30["hard_hit_rate"],
                    "Avg EV": b30["ev"],
                    "Pitcher HR/PA": p90["hr_rate"],
                    "Expected PA": result["expected_pa"],
                    "Lineup": "Confirmed" if confirmed else "Lineup pending",
                    "Batting Order": slot if confirmed else np.nan,
                    "Reason": "",
                    "Data Confidence": result["data_confidence"],
                    "Pick Type": "",
                })

            completed += 1
            elapsed = time.perf_counter() - started_at
            progress.progress(
                completed / total_steps,
                text=f"⚡ Analyzing slate… {completed}/{total_steps} team sides · {elapsed:.0f}s",
            )

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        st.warning("No hitters could be evaluated.")
        st.stop()

    result_df = result_df.sort_values(
        ["HR Probability", "Score", "Data Confidence"],
        ascending=False,
    ).reset_index(drop=True)

    elapsed = time.perf_counter() - started_at
    progress.progress(
        1.0,
        text=f"✅ Ready — {len(result_df)} hitters analyzed in {elapsed:.1f}s",
    )
    st.caption(
        f"⚡ v10.1 speed layer: {len(unique_hitter_ids)} unique hitters · "
        f"{len(unique_pitcher_ids)} unique pitchers · {elapsed:.1f}s total. "
        f"Each hitter's 90-day Statcast data is cached and reused for recent + BvP metrics."
    )
    result_df["Reason"] = result_df.apply(reason, axis=1)
    result_df["Pick Type"] = [pick_label(row, i + 1, 0) for i, (_, row) in enumerate(result_df.iterrows())]

    best = result_df.iloc[0]
    st.subheader(f"🏆 Top {top_n} — {selected_date.strftime('%B %-d, %Y')}")
    st.caption(
        f"Data window: {recent_days} days. Lineups are treated as opportunity information. "
        f"Last refreshed: {pd.Timestamp.now().strftime('%I:%M %p')}"
    )

    a, b, c, d = st.columns(4)
    with a:
        st.metric("🥇 Best Pick", best["Player"])
    with b:
        st.metric("HR Probability", f'{best["HR Probability"]:.1%}')
    with c:
        st.metric("Model Score", f'{best["Score"]:.0f}/100')
    with d:
        st.metric("Confidence", best["Confidence"])

    if best["HR Probability"] >= 0.35:
        st.warning("⚠️ Model Alert: this is an unusually high HR probability. Verify the lineup, pitcher and underlying Statcast sample before treating it as a premium pick.")
    elif best["HR Probability"] >= 0.25:
        st.info("🔥 High-end model projection. The probability is an estimate, not a guarantee.")

    st.subheader("🔥 Top Picks")
    for rank, (_, pick) in enumerate(result_df.head(top_n).iterrows(), start=1):
        confidence_badge = {"HIGH": "🟢 HIGH", "GOOD": "🔵 GOOD", "WATCH": "🟡 WATCH"}.get(pick["Confidence"], pick["Confidence"])
        lineup_text = (
            f'Confirmed #{int(pick["Batting Order"])}' if pick["Lineup"] == "Confirmed" and pd.notna(pick["Batting Order"])
            else "Lineup pending"
        )
        st.markdown(
            f'''<div class="pick-card">
            <div style="display:flex;justify-content:space-between;gap:12px;align-items:flex-start;">
              <div><div style="font-size:0.9rem;opacity:.75;">#{rank} · {pick["Pick Type"]}</div>
              <div style="font-size:1.35rem;font-weight:700;">{pick["Player"]}</div>
              <div style="opacity:.8;">{pick["Team"]} vs {pick["Opponent"]} · {pick["Opposing SP"]}</div></div>
              <div style="text-align:right;"><div style="font-size:1.8rem;font-weight:800;">{pick["HR Probability"]:.1%}</div>
              <div style="opacity:.75;">HR probability</div></div>
            </div>
            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:14px;">
              <div><b>{pick["Score"]:.0f}</b><br><span style="opacity:.7;">Score</span></div>
              <div><b>{pick["Barrel %"]:.1%}</b><br><span style="opacity:.7;">Barrel</span></div>
              <div><b>{pick["Hard-Hit %"]:.1%}</b><br><span style="opacity:.7;">Hard-Hit</span></div>
              <div><b>{pick["Avg EV"]:.1f}</b><br><span style="opacity:.7;">Avg EV</span></div>
            </div>
            <div style="margin-top:10px;opacity:.8;">{confidence_badge} · {lineup_text} · {pick["Venue"]}</div>
            </div>''',
            unsafe_allow_html=True,
        )
        with st.expander(f"🔎 Why {pick['Player']}?"):
            st.write(pick["Reason"].capitalize() + ".")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("HR/PA", f'{pick["HR/PA"]:.3f}')
            r2.metric("Pitcher HR/PA", f'{pick["Pitcher HR/PA"]:.3f}')
            r3.metric("Expected PA", f'{pick["Expected PA"]:.1f}')
            r4.metric("BvP PA", f'{int(pick["Matchup PA"])}')
            st.caption(
                f'Weather: {pick["Weather"]}'
                + (f' · {pick["Temp F"]:.0f}°F' if pd.notna(pick["Temp F"]) else "")
                + (f' · wind {pick["Wind MPH"]:.0f} mph' if pd.notna(pick["Wind MPH"]) else "")
            )

    st.subheader("🎯 Pick Types")
    confirmed_top = result_df[result_df["Lineup"] == "Confirmed"].head(15)
    if not confirmed_top.empty:
        value = confirmed_top.sort_values(["Score", "HR Probability"], ascending=False).iloc[0]
        sleeper_candidates = confirmed_top[confirmed_top["Barrel %"] >= 0.08]
        sleeper = sleeper_candidates.sort_values("HR Probability", ascending=False).iloc[0] if not sleeper_candidates.empty else value
        x, y = st.columns(2)
        with x:
            st.info(f'💎 **Value Pick:** {value["Player"]} — {value["HR Probability"]:.1%} HR probability, {value["Score"]:.0f}/100 score.')
        with y:
            st.info(f'🌙 **Sleeper Pick:** {sleeper["Player"]} — {sleeper["HR Probability"]:.1%} HR probability, {sleeper["Barrel %"]:.1%} barrels.')

    st.subheader("📊 Detailed Rankings")
    display = result_df.head(top_n).copy()
    display["HR Probability"] = display["HR Probability"].map(lambda x: f"{x:.1%}")
    display["Score"] = display["Score"].map(lambda x: f"{x:.0f}")
    display["HR/PA"] = display["HR/PA"].map(lambda x: f"{x:.3f}")
    display["Barrel %"] = display["Barrel %"].map(lambda x: f"{x:.1%}")
    display["Hard-Hit %"] = display["Hard-Hit %"].map(lambda x: f"{x:.1%}")
    display["Pitcher HR/PA"] = display["Pitcher HR/PA"].map(lambda x: f"{x:.3f}")
    display["Avg EV"] = display["Avg EV"].map(lambda x: f"{x:.1f}")
    display["Expected PA"] = display["Expected PA"].map(lambda x: f"{x:.1f}")
    display["Matchup HR/PA"] = display["Matchup HR/PA"].map(lambda x: f"{x:.3f}")

    columns = [
        "Player", "Team", "Opponent", "Opposing SP", "HR Probability", "Score",
        "Confidence", "Pick Type", "Lineup", "Barrel %", "Hard-Hit %", "Avg EV",
        "Pitcher HR/PA", "Matchup HR/PA", "Expected PA", "Weather", "Venue",
    ]
    st.dataframe(display[columns], use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download today's rankings as CSV",
        result_df.to_csv(index=False).encode("utf-8"),
        file_name=f"mlb_hr_predictions_{selected_date.isoformat()}.csv",
        mime="text/csv",
    )

    st.divider()
    with st.expander("📘 How the model works"):
        st.write(
            "The model combines recent and 90-day HR/PA, barrel rate, hard-hit rate, "
            "average exit velocity, opposing starter HR/PA, batter-vs-pitcher history, "
            "weather and expected plate appearances. Small samples are shrunk toward "
            "league-average priors."
        )
        st.write(
            "HR Probability is the primary ranking signal. Model Score is a normalized "
            "0–100 index based on the probability, so it is deliberately more spread out "
            "than the previous saturated score."
        )
        st.write(
            "Confidence measures data quality and lineup certainty; it is not the same thing "
            "as the probability of a home run."
        )
        st.caption("Statcast data: MLB StatsAPI + Baseball Savant/pybaseball. Weather: Open-Meteo when available.")

st.caption("MLB HR Predictor v10.1 • Safe speed layer • cached Statcast reuse")
