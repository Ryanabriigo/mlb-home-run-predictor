
import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import date, timedelta
import math
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
# Statcast helpers
# -------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def get_batter_statcast(player_id, start_date, end_date):
    """Fetch one 90-day Statcast window per hitter and cache the raw data.
    All shorter-window metrics are calculated locally from this cached dataset.
    """
    try:
        df = statcast_batter(start_date, end_date, player_id)
        if df is None:
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()


def summarize_batter(df, window_start=None):
    if df is None or df.empty:
        return {"pa": 0, "hr": 0, "hr_pa": LEAGUE_HR_PA,
                "barrel_rate": DEFAULT_BARREL, "hard_hit_rate": DEFAULT_HARD_HIT,
                "ev": DEFAULT_EV, "bbe": 0}
    x = df.copy()
    if window_start is not None and "game_date" in x.columns:
        x = x[pd.to_datetime(x["game_date"], errors="coerce").dt.date >= window_start]
    pa_df = x[x["events"].notna()].copy()
    bbe = x[x["launch_speed"].notna()].copy()
    pa = len(pa_df)
    hr = int((pa_df["events"] == "home_run").sum())
    if len(bbe):
        hard_hit = float((bbe["launch_speed"] >= 95).mean())
        ev = float(bbe["launch_speed"].mean())
        barrel_rate = float((bbe["launch_speed_angle"] == 6).mean()) if "launch_speed_angle" in bbe.columns else DEFAULT_BARREL
    else:
        hard_hit, ev, barrel_rate = DEFAULT_HARD_HIT, DEFAULT_EV, DEFAULT_BARREL
    reliability = min(1.0, pa / 120.0)
    return {
        "pa": pa, "hr": hr,
        "hr_pa": float(reliability * (hr / max(pa, 1)) + (1 - reliability) * LEAGUE_HR_PA),
        "barrel_rate": float(reliability * barrel_rate + (1 - reliability) * DEFAULT_BARREL),
        "hard_hit_rate": float(reliability * hard_hit + (1 - reliability) * DEFAULT_HARD_HIT),
        "ev": float(reliability * ev + (1 - reliability) * DEFAULT_EV),
        "bbe": len(bbe),
    }


def get_batter_windows(player_id, recent_start, long_start, end_date):
    """Single network request; derive both recent and long windows locally."""
    raw = get_batter_statcast(player_id, long_start, end_date)
    return summarize_batter(raw, recent_start), summarize_batter(raw, None)


@st.cache_data(ttl=1800, show_spinner=False)
def get_pitcher_statcast(player_id, start_date, end_date):
    if not player_id:
        return pd.DataFrame()
    try:
        df = statcast_pitcher(start_date, end_date, player_id)
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def summarize_pitcher(df, batter_id=None):
    if df is None or df.empty:
        return {"hr_rate": LEAGUE_HR_PA, "hr": 0, "pa": 0}
    x = df
    if batter_id is not None and "batter" in x.columns:
        x = x[x["batter"] == batter_id]
    pa_df = x[x["events"].notna()].copy()
    pa = len(pa_df)
    hr = int((pa_df["events"] == "home_run").sum())
    raw = hr / max(pa, 1)
    reliability = min(1.0, pa / 180.0)
    return {"hr_rate": float(reliability * raw + (1 - reliability) * LEAGUE_HR_PA), "hr": hr, "pa": pa}


def get_batter_pitcher_matchup_from_df(pitcher_df, batter_id):
    return summarize_pitcher(pitcher_df, batter_id)


@st.cache_data(ttl=900, show_spinner=False)
def get_weather(lat, lon, game_date, game_time=None):
    if lat is None or lon is None:
        return {"condition": "Unknown", "temp_f": np.nan, "wind_mph": np.nan, "multiplier": 1.0}
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "hourly": "temperature_2m,wind_speed_10m,weather_code",
                    "temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "forecast_days": 1,
                    "timezone": "auto"}, timeout=8)
        r.raise_for_status()
        data = r.json()
        temps = data.get("hourly", {}).get("temperature_2m", [])
        winds = data.get("hourly", {}).get("wind_speed_10m", [])
        temp = float(np.nanmean(temps)) if temps else np.nan
        wind = float(np.nanmean(winds)) if winds else np.nan
        return {"condition": "Forecast", "temp_f": temp, "wind_mph": wind, "multiplier": 1.0}
    except Exception:
        return {"condition": "Unknown", "temp_f": np.nan, "wind_mph": np.nan, "multiplier": 1.0}


def weather_multiplier(weather):
    return float(weather.get("multiplier", 1.0)) if weather else 1.0


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

    raw_score = 100 * clamp(
        0.38 * min(hitter_quality / 2.5, 1)
        + 0.22 * min(pitcher_quality / 1.6, 1)
        + 0.18 * min(barrel_signal / 2.0, 1)
        + 0.12 * min(hard_hit_signal / 1.5, 1)
        + 0.10 * min(expected_pa / 4.5, 1),
        0, 1
    )
    score = raw_score * lineup_penalty

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
    "A Statcast-based estimate of each hitter's chance to hit at least one home run. "
    "Probabilities are estimates, not guarantees."
)

col1, col2, col3 = st.columns(3)
with col1:
    selected_date = st.date_input("Game date", date.today())
with col2:
    recent_days = st.slider("Recent window", 7, 45, 30)
with col3:
    top_n = st.slider("Top picks", 5, 25, 10)

run = st.button(
    "🚀 RUN TODAY'S HR PICKS",
    type="primary",
    use_container_width=True,
)

st.caption("⚡ Fast mode: Statcast datasets are cached and reused across reruns. Change settings, then press Run to refresh predictions.")
st.divider()
with st.expander("📚 Prediction History & Model Performance"):
    history = load_prediction_history()
    summary = performance_summary(history)
    if summary is None:
        st.info("No completed predictions have been tracked yet. Run predictions each day and record results to build the model history.")
    else:
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Predictions Checked", summary["checked"])
        h2.metric("Top Pick HR Rate", f'{summary["top1"]:.1%}')
        h3.metric("Top 5 HR Rate", f'{summary["top5"]:.1%}')
        h4.metric("Brier Score", f'{summary["brier"]:.3f}')
        st.dataframe(history.tail(100), use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download prediction history",
            history.to_csv(index=False).encode("utf-8"),
            file_name="mlb_hr_prediction_history.csv",
            mime="text/csv",
        )

if run:
    st.session_state["prediction_key"] = f"{selected_date.isoformat()}_{recent_days}"
    games_df = get_games(selected_date.isoformat())

    if games_df.empty:
        st.warning("No MLB games were found for this date.")
        st.stop()

    season = selected_date.year
    start_30 = selected_date - timedelta(days=recent_days)
    start_90 = selected_date - timedelta(days=90)

    rows = []
    progress = st.progress(0)
    total_steps = max(len(games_df) * 2, 1)
    completed = 0

    for _, game in games_df.iterrows():
        for side in ("away", "home"):
            team = game[side]
            team_id = int(game[f"{side}_id"])

            opponent = game["home"] if side == "away" else game["away"]
            pitcher = game["home_sp"] if side == "away" else game["away_sp"]

            pitcher_id = pitcher.get("id")
            pitcher_name = pitcher.get("fullName", "TBD")

            pitcher_raw = get_pitcher_statcast(
                pitcher_id, start_90.isoformat(), selected_date.isoformat()
            )
            p90 = summarize_pitcher(pitcher_raw)

            weather = get_weather(
                game.get("venue_lat"), game.get("venue_lon"),
                selected_date.isoformat(), game.get("game_time")
            )

            lineup = get_lineup(int(game["game_pk"]), side)
            confirmed = not lineup.empty

            hitters = get_team_hitters(team_id, season)

            if confirmed:
                hitters = hitters.merge(
                    lineup[["id", "order"]],
                    on="id",
                    how="inner",
                )
                hitters["order"] = hitters["order"].astype(int)
            else:
                hitters["order"] = np.nan
                # Pending lineups: limit expensive Statcast work to a compact likely-hitter pool.
                # Once MLB posts the lineup, the confirmed nine are analyzed instead.
                hitters = hitters.head(12).copy()

            for _, hitter in hitters.iterrows():
                pid = int(hitter["id"])

                # One hitter Statcast request supplies both recent and 90-day metrics.
                b30, b90 = get_batter_windows(
                    pid, start_30, start_90.isoformat(), selected_date.isoformat()
                )

                # Reuse the already-cached pitcher dataset for BvP instead of another request.
                matchup = get_batter_pitcher_matchup_from_df(pitcher_raw, pid)

                result = model_player(
                    b30,
                    b90,
                    p90,
                    confirmed,
                    hitter["order"] if pd.notna(hitter["order"]) else 5,
                    matchup=matchup,
                    weather=weather,
                )

                rows.append(
                    {
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
                        "Batting Order": hitter["order"] if pd.notna(hitter["order"]) else np.nan,
                        "Reason": "",
                        "Lineup Penalty": result["lineup_penalty"],
                        "Data Confidence": result["data_confidence"],
                        "Pick Type": "",
                    }
                )

            completed += 1
            progress.progress(completed / total_steps)

    result_df = pd.DataFrame(rows)

    if result_df.empty:
        st.warning("No hitters could be evaluated.")
        st.stop()

    # Retain only the top 10 from each run for later result tracking.
    # This keeps session memory small while allowing the app to learn its recent hit rate.
    history_rows = result_df.head(10).copy()
    for _, h in history_rows.iterrows():
        st.session_state.setdefault("prediction_history", []).append({
            "date": selected_date.isoformat(),
            "game_pk": int(h["Game PK"]),
            "player_id": int(h["Player ID"]),
            "player": h["Player"],
            "rank": int(_ + 1),
            "probability": float(h["HR Probability"]),
        })

    result_df = result_df.sort_values(
        ["HR Probability", "Score", "Data Confidence" if "Data Confidence" in result_df.columns else "Score"],
        ascending=False,
    ).reset_index(drop=True)

    result_df["Reason"] = result_df.apply(reason, axis=1)
    result_df["Pick Type"] = [pick_label(row, i + 1, 0) for i, (_, row) in enumerate(result_df.iterrows())]

    # -------------------------
    # Daily board
    # -------------------------
    best = result_df.iloc[0]
    st.subheader("🏆 Today's HR Board")
    st.caption("The model explicitly discounts hitters whose starting lineup has not been confirmed.")

    a, b, c, d = st.columns(4)
    with a:
        st.metric("🥇 Best Pick", best["Player"])
    with b:
        st.metric("HR Probability", f'{best["HR Probability"]:.1%}')
    with c:
        st.metric("Model Score", f'{best["Score"]:.0f}/100')
    with d:
        st.metric("Confidence", best["Confidence"])

    st.success(
        f'**{best["Player"]}** — {best["Team"]} vs {best["Opponent"]}. '
        f'{best["Reason"].capitalize()}.'
    )

    st.subheader("🔥 Top 5 Picks")
    top5 = result_df.head(5).copy()
    cards = st.columns(5)
    for rank, (_, pick) in enumerate(top5.iterrows(), start=1):
        with cards[rank - 1]:
            st.markdown(f"### #{rank}")
            st.markdown(f"**{pick['Player']}**")
            st.metric("HR", f"{pick['HR Probability']:.1%}")
            st.metric("Score", f"{pick['Score']:.0f}/100")
            st.caption(f"{pick['Pick Type']}\n\n{pick['Team']} vs {pick['Opponent']}")
            st.caption(pick["Reason"].capitalize())

    # -------------------------
    # Pick types
    # -------------------------
    st.subheader("🎯 Pick Types")
    confirmed_top = result_df[result_df["Lineup"] == "Confirmed"].head(10)
    if not confirmed_top.empty:
        value = confirmed_top.sort_values(["Score", "HR Probability"], ascending=False).iloc[0]
        sleeper_candidates = confirmed_top[confirmed_top["Barrel %"] >= 0.08]
        sleeper = sleeper_candidates.sort_values("HR Probability", ascending=False).iloc[0] if not sleeper_candidates.empty else value
        x, y = st.columns(2)
        with x:
            st.info(f"💎 **Value Pick:** {value['Player']} — {value['HR Probability']:.1%} HR probability, {value['Score']:.0f}/100 score.")
        with y:
            st.info(f"🌙 **Sleeper Pick:** {sleeper['Player']} — {sleeper['HR Probability']:.1%} HR probability, {sleeper['Barrel %']:.1%} barrels.")

    # -------------------------
    # Full rankings table
    # -------------------------
    st.subheader(f"📊 Full Top {top_n} HR Rankings")

    display = result_df.head(top_n).copy()

    display["HR Probability"] = display["HR Probability"].map(lambda x: f"{x:.1%}")
    display["Score"] = display["Score"].map(lambda x: f"{x:.0f}")
    display["HR/PA"] = display["HR/PA"].map(lambda x: f"{x:.3f}")
    display["Barrel %"] = display["Barrel %"].map(lambda x: f"{x:.1%}")
    display["Hard-Hit %"] = display["Hard-Hit %"].map(lambda x: f"{x:.1%}")
    display["Pitcher HR/PA"] = display["Pitcher HR/PA"].map(lambda x: f"{x:.3f}")
    display["Matchup HR/PA"] = display["Matchup HR/PA"].map(lambda x: f"{x:.3f}")
    display["Temp F"] = display["Temp F"].map(lambda x: f"{x:.0f}" if pd.notna(x) else "—")
    display["Wind MPH"] = display["Wind MPH"].map(lambda x: f"{x:.0f}" if pd.notna(x) else "—")
    display["Avg EV"] = display["Avg EV"].map(lambda x: f"{x:.1f}")
    display["Expected PA"] = display["Expected PA"].map(lambda x: f"{x:.1f}")

    columns = [
        "Player", "Team", "Opponent", "Opposing SP",
        "HR Probability", "Score", "Confidence", "Pick Type",
        "Lineup", "Barrel %", "Hard-Hit %", "Avg EV",
        "Pitcher HR/PA", "Matchup HR/PA", "Expected PA", "Weather", "Temp F", "Wind MPH", "Venue",
    ]

    st.dataframe(
        display[columns],
        use_container_width=True,
        hide_index=True,
    )

    # -------------------------
    # Why the picks rank highly
    # -------------------------
    st.subheader("🔎 Why the top picks rank highly")

    for i, row in result_df.head(min(5, top_n)).iterrows():
        st.write(
            f"**#{i+1} {row['Player']} — {row['HR Probability']:.1%}**  \n"
            f"{row['Reason'].capitalize()}."
        )

    st.caption(
        "Best practice: run the model again after official batting orders are posted. "
        "The model treats lineup confirmation and batting-order position as opportunity information."
    )

    # -------------------------
    # Recent model performance
    # -------------------------
    st.subheader("📈 Model Performance")
    hist = load_prediction_history()
    if hist is not None and not hist.empty:
        checked = hist[hist["actual_hr"].notna()].copy()
        if not checked.empty:
            top5 = checked[checked["rank"] <= 5]
            top1 = checked[checked["rank"] == 1]
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Top Pick HR Rate", f'{top1["actual_hr"].gt(0).mean():.1%}' if not top1.empty else "—")
            with m2:
                st.metric("Top 5 HR Rate", f'{top5["actual_hr"].gt(0).mean():.1%}' if not top5.empty else "—")
            with m3:
                st.metric("Predictions Checked", f'{len(checked)}')
            st.caption("These are simple hit rates from predictions retained in this browser session. They are not a full historical backtest.")
            st.dataframe(
                checked[["date","rank","player","probability","actual_hr"]]
                .sort_values(["date","rank"], ascending=[False, True])
                .head(50),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Predictions are saved for this session. Reopen the app after games finish and use the same session to check results.")
    else:
        st.info("Run predictions to begin building a performance history.")

    st.download_button(
        "⬇️ Download today's rankings as CSV",
        result_df.to_csv(index=False).encode("utf-8"),
        file_name=f"mlb_hr_predictions_{selected_date.isoformat()}.csv",
        mime="text/csv",
    )

st.divider()

with st.expander("How the model works"):
    st.write(
        "The model combines recent and 90-day HR/PA, barrel rate, hard-hit rate, "
        "average exit velocity, opposing starter HR/PA, and expected plate appearances. "
        "Small samples are shrunk toward league-average priors. A small, heavily shrunk "
        "batter-vs-pitcher matchup signal and conservative weather adjustment are also used."
    )
    st.write(
        "Statcast defines a hard-hit ball as 95+ mph and defines a barrel using the "
        "combination of exit velocity and launch angle. These are useful quality-of-contact "
        "inputs for HR modeling."
    )
    st.write(
        "The displayed HR probability is an estimate of at least one HR in the game; "
        "it is not a sportsbook probability and should not be interpreted as a guarantee."
    )

st.caption("MLB HR Predictor v8 • Fast Statcast Engine • Data: MLB StatsAPI + Baseball Savant/Statcast via pybaseball")
