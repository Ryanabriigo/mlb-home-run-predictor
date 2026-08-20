
import streamlit as st
import requests
import pandas as pd
import numpy as np
import io
from datetime import date, timedelta
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

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
# Fast daily data layer
# -------------------------
SAVANT = "https://baseballsavant.mlb.com"

@st.cache_data(ttl=21600, show_spinner=False)
def savant_csv(start_date, end_date, player_type):
    """Download a compact all-MLB Statcast window.

    Savant is much faster when we ask for the whole league once instead of
    making one request per hitter/pitcher. The date window is kept <= 5 days
    per request because Savant can truncate larger pitch-level exports.
    """
    url = f"{SAVANT}/statcast_search/csv"
    params = {
        "all": "true",
        "hfPT": "", "hfAB": "", "hfBBT": "", "hfPR": "", "hfZ": "",
        "stadium": "", "hfBBL": "", "hfNewZones": "", "hfGT": "R|",
        "hfSea": "", "hfSit": "", "player_type": player_type,
        "hfOuts": "", "opponent": "", "pitcher_throws": "",
        "batter_stands": "", "hfSA": "", "game_date_gt": start_date,
        "game_date_lt": end_date, "team": "", "position": "",
        "hfRO": "", "home_road": "", "hfFlag": "", "hfPull": "",
        "metric_1": "", "hfInn": "", "min_pitches": 0,
        "min_results": 0, "group_by": "name", "sort_col": "pitches",
        "player_event_sort": "h_launch_speed", "sort_order": "desc",
        "min_abs": 0, "type": "details",
    }
    r = requests.get(url, params=params, timeout=90)
    r.raise_for_status()
    text = r.content.decode("utf-8-sig", errors="replace")
    if not text.strip() or text.lstrip().startswith("<"):
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(text), low_memory=False)


def _five_day_ranges(start_date, end_date):
    cur = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    ranges = []
    while cur <= end:
        chunk_end = min(cur + timedelta(days=4), end)
        ranges.append((cur.isoformat(), chunk_end.isoformat()))
        cur = chunk_end + timedelta(days=1)
    return ranges


@st.cache_data(ttl=21600, show_spinner=False)
def get_daily_statcast_layer(start_date, end_date, player_type):
    """Build one cached all-MLB Statcast layer.

    Savant is split into <=5-day chunks, but those chunks are independent,
    so fetch them concurrently. This reduces cold-cache network wait while
    keeping the same data/model inputs.
    """
    ranges = _five_day_ranges(start_date, end_date)
    if not ranges:
        return pd.DataFrame()

    frames = []
    with ThreadPoolExecutor(max_workers=min(4, len(ranges))) as pool:
        futures = {
            pool.submit(savant_csv, a, b, player_type): (a, b)
            for a, b in ranges
        }
        for fut in as_completed(futures):
            try:
                df = fut.result()
                if df is not None and not df.empty:
                    frames.append(df)
            except Exception:
                continue

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date
    return df


@st.cache_data(ttl=21600, show_spinner=False)
def get_savant_season_leaderboard(season):
    """Compact season-to-date contact-quality leaderboard (one request)."""
    try:
        r = requests.get(
            f"{SAVANT}/leaderboard/statcast",
            params={"type": "batter", "year": season, "csv": "true"},
            timeout=45,
        )
        r.raise_for_status()
        text = r.content.decode("utf-8-sig", errors="replace")
        return pd.read_csv(io.StringIO(text), low_memory=False)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def get_season_batting_stats(season):
    """One MLB StatsAPI request for season HR/PA by hitter."""
    try:
        data = api_get("stats", {
            "stats": "season", "group": "hitting", "season": season,
            "sportIds": 1, "limit": 1000,
        })
        rows = []
        for split in data.get("stats", [{}])[0].get("splits", []):
            stat = split.get("stat", {})
            person = split.get("player", {})
            rows.append({
                "player_id": int(person.get("id")),
                "pa": float(stat.get("plateAppearances", 0) or 0),
                "hr": float(stat.get("homeRuns", 0) or 0),
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def get_season_pitching_stats(season):
    """One MLB StatsAPI request for season HR/BF by pitcher."""
    try:
        data = api_get("stats", {
            "stats": "season", "group": "pitching", "season": season,
            "sportIds": 1, "limit": 1000,
        })
        rows = []
        for split in data.get("stats", [{}])[0].get("splits", []):
            stat = split.get("stat", {})
            person = split.get("player", {})
            rows.append({
                "player_id": int(person.get("id")),
                "bf": float(stat.get("battersFaced", 0) or 0),
                "hr": float(stat.get("homeRuns", 0) or 0),
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


def _num(row, *names, default=np.nan):
    for name in names:
        if name in row.index and pd.notna(row[name]):
            try:
                return float(row[name])
            except Exception:
                pass
    return default


def build_season_batter_map(season):
    leaderboard = get_savant_season_leaderboard(season)
    batting = get_season_batting_stats(season)
    out = {}
    if leaderboard is not None and not leaderboard.empty:
        pid_col = next((c for c in ["player_id", "playerid"] if c in leaderboard.columns), None)
        if pid_col:
            for _, r in leaderboard.iterrows():
                pid = int(r[pid_col])
                bbe = _num(r, "attempts", "bbe", default=0)
                pa = _num(r, "brl_pa", default=np.nan)
                barrels_pa_pct = _num(r, "brl_pa", default=np.nan)
                # Savant's brl_pa is percent, so convert to decimal.
                barrel_rate = barrels_pa_pct / 100.0 if pd.notna(barrels_pa_pct) else DEFAULT_BARREL
                hard_pct = _num(r, "ev95percent", "ev95_pct", default=DEFAULT_HARD_HIT * 100) / 100.0
                ev = _num(r, "avg_hit_speed", "avg_ev", default=DEFAULT_EV)
                out[pid] = {
                    "pa": 0, "hr": 0, "hr_pa": LEAGUE_HR_PA,
                    "barrel_rate": barrel_rate, "hard_hit_rate": hard_pct,
                    "ev": ev, "bbe": bbe,
                }
    if batting is not None and not batting.empty:
        for _, r in batting.iterrows():
            pid = int(r["player_id"])
            pa = float(r["pa"])
            hr = float(r["hr"])
            base = out.setdefault(pid, {
                "pa": 0, "hr": 0, "hr_pa": LEAGUE_HR_PA,
                "barrel_rate": DEFAULT_BARREL, "hard_hit_rate": DEFAULT_HARD_HIT,
                "ev": DEFAULT_EV, "bbe": 0,
            })
            rel = min(1.0, pa / 220.0)
            base["pa"] = pa
            base["hr"] = hr
            base["hr_pa"] = float(rel * (hr / max(pa, 1)) + (1-rel) * LEAGUE_HR_PA)
    return out


def build_pitcher_map(season):
    pitching = get_season_pitching_stats(season)
    out = {}
    if pitching is not None and not pitching.empty:
        for _, r in pitching.iterrows():
            bf = float(r["bf"])
            hr = float(r["hr"])
            rel = min(1.0, bf / 450.0)
            raw = hr / max(bf, 1)
            out[int(r["player_id"])] = {
                "hr_rate": float(rel * raw + (1-rel) * LEAGUE_HR_PA),
                "hr": hr, "pa": bf,
            }
    return out


@st.cache_data(ttl=21600, show_spinner=False)
def build_season_baselines_fast(season):
    """Build hitter and pitcher baselines concurrently on a cold cache."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        batter_future = pool.submit(build_season_batter_map, season)
        pitcher_future = pool.submit(build_pitcher_map, season)
        return batter_future.result(), pitcher_future.result()


def summarize_recent_all(df, player_id, start_date):
    if df is None or df.empty:
        return summarize_batter(pd.DataFrame())
    x = df[df.get("batter", pd.Series(dtype=float)).eq(player_id)].copy()
    return summarize_batter(x, start_date)


def summarize_recent_pitcher(df, pitcher_id, batter_id=None):
    if df is None or df.empty or "pitcher" not in df.columns:
        return {"hr_rate": LEAGUE_HR_PA, "hr": 0, "pa": 0}
    x = df[df["pitcher"].eq(pitcher_id)].copy()
    if batter_id is not None and "batter" in x.columns:
        x = x[x["batter"].eq(batter_id)]
    pa_df = x[x["events"].notna()].copy() if "events" in x.columns else x.iloc[0:0]
    pa = len(pa_df)
    hr = int((pa_df["events"] == "home_run").sum()) if pa else 0
    raw = hr / max(pa, 1)
    reliability = min(1.0, pa / 60.0)
    return {"hr_rate": float(reliability * raw + (1-reliability) * LEAGUE_HR_PA), "hr": hr, "pa": pa}


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
# Parallel prefetch helpers
# -------------------------
PREFETCH_WORKERS = 8

def parallel_map(items, func, workers=PREFETCH_WORKERS):
    """Run independent network-bound jobs concurrently while preserving keys."""
    if not items:
        return {}
    out = {}
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as pool:
        futures = {pool.submit(func, item): item for item in items}
        for fut in as_completed(futures):
            item = futures[fut]
            try:
                out[item] = fut.result()
            except Exception:
                out[item] = None
    return out

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

st.caption("💾 Daily results are cached by date + lookback window so the app avoids rebuilding the same prediction table.")

st.caption("⚡ v11 Fast Cache: parallel Statcast downloads + cached daily predictions. First run builds the day; repeat runs are designed to be near-instant.")
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

@st.cache_data(persist="disk", show_spinner=False)
def build_daily_predictions(selected_date_iso, recent_days):
    """Compute and cache the complete daily prediction table.

    The completed table is cached by date/lookback, so the same prediction
    does not have to be rebuilt on every Streamlit rerun.
    """
    selected = pd.Timestamp(selected_date_iso).date()
    season = selected.year
    start_recent = selected - timedelta(days=recent_days)
    end_date = selected.isoformat()

    games_df = get_games(end_date)
    if games_df.empty:
        return pd.DataFrame()

    game_records = games_df.to_dict("records")

    lineup_jobs = [
        (int(g["game_pk"]), side)
        for g in game_records
        for side in ("away", "home")
    ]
    lineup_map = parallel_map(
        lineup_jobs,
        lambda key: get_lineup(key[0], key[1]),
        workers=8,
    )

    roster_jobs = [
        (int(g[f"{side}_id"]), season)
        for g in game_records
        for side in ("away", "home")
    ]
    roster_map = parallel_map(
        roster_jobs,
        lambda key: get_team_hitters(key[0], key[1]),
        workers=8,
    )

    candidate_specs = []
    unique_weather = {}

    for g in game_records:
        game_pk = int(g["game_pk"])
        unique_weather[game_pk] = (
            g.get("venue_lat"),
            g.get("venue_lon"),
            end_date,
            g.get("game_time"),
        )

        for side in ("away", "home"):
            team = g[side]
            team_id = int(g[f"{side}_id"])
            opponent = g["home"] if side == "away" else g["away"]
            pitcher = g["home_sp"] if side == "away" else g["away_sp"]
            pitcher_id = pitcher.get("id")
            pitcher_name = pitcher.get("fullName", "TBD")

            lineup = lineup_map.get((game_pk, side))
            if lineup is None:
                lineup = pd.DataFrame(columns=["id", "name", "order"])

            confirmed = not lineup.empty
            hitters = roster_map.get((team_id, season))
            if hitters is None:
                hitters = pd.DataFrame(columns=["id", "name", "position"])
            hitters = hitters.copy()

            if confirmed:
                hitters = hitters.merge(
                    lineup[["id", "order"]],
                    on="id",
                    how="inner",
                )
                hitters["order"] = hitters["order"].astype(int)
            else:
                hitters["order"] = np.nan
                hitters = hitters.head(8).copy()

            for _, h in hitters.iterrows():
                candidate_specs.append({
                    "game": g,
                    "side": side,
                    "team": team,
                    "team_id": team_id,
                    "opponent": opponent,
                    "pitcher_id": int(pitcher_id) if pitcher_id else None,
                    "pitcher_name": pitcher_name,
                    "confirmed": confirmed,
                    "pid": int(h["id"]),
                    "player": h["name"],
                    "order": h["order"],
                })

    recent_layer = get_daily_statcast_layer(
        start_recent.isoformat(),
        end_date,
        "batter",
    )

    season_batter_map, season_pitcher_map = build_season_baselines_fast(season)

    weather_map = parallel_map(
        list(unique_weather.keys()),
        lambda game_pk: get_weather(*unique_weather[game_pk]),
        workers=8,
    )

    rows = []
    default_batter = {
        "pa": 0,
        "hr": 0,
        "hr_pa": LEAGUE_HR_PA,
        "barrel_rate": DEFAULT_BARREL,
        "hard_hit_rate": DEFAULT_HARD_HIT,
        "ev": DEFAULT_EV,
        "bbe": 0,
    }

    for c in candidate_specs:
        g = c["game"]
        b_recent = summarize_recent_all(
            recent_layer,
            c["pid"],
            start_recent,
        )
        b_long = season_batter_map.get(c["pid"], default_batter)

        if c["pitcher_id"]:
            p90 = season_pitcher_map.get(
                c["pitcher_id"],
                {"hr_rate": LEAGUE_HR_PA, "hr": 0, "pa": 0},
            )
            matchup = summarize_recent_pitcher(
                recent_layer,
                c["pitcher_id"],
                c["pid"],
            )
        else:
            p90 = {"hr_rate": LEAGUE_HR_PA, "hr": 0, "pa": 0}
            matchup = {"hr_rate": LEAGUE_HR_PA, "hr": 0, "pa": 0}

        matchup["hr_pa"] = matchup.get("hr_rate", LEAGUE_HR_PA)
        weather = weather_map.get(
            int(g["game_pk"]),
            {
                "condition": "Unknown",
                "temp_f": np.nan,
                "wind_mph": np.nan,
                "multiplier": 1.0,
            },
        )

        result = model_player(
            b_recent,
            b_long,
            p90,
            c["confirmed"],
            c["order"] if pd.notna(c["order"]) else 5,
            matchup=matchup,
            weather=weather,
        )

        rows.append({
            "Player": c["player"],
            "Team": c["team"],
            "Opponent": c["opponent"],
            "Opposing SP": c["pitcher_name"],
            "Venue": g["venue"],
            "Game PK": int(g["game_pk"]),
            "Player ID": c["pid"],
            "Weather": weather.get("condition", "Unknown"),
            "Temp F": weather.get("temp_f"),
            "Wind MPH": weather.get("wind_mph"),
            "Matchup PA": matchup.get("pa", 0),
            "Matchup HR/PA": matchup.get("hr_pa", LEAGUE_HR_PA),
            "HR Probability": result["probability"],
            "Score": result["score"],
            "Confidence": result["confidence"],
            "HR/PA": b_recent["hr_pa"],
            "Barrel %": b_recent["barrel_rate"],
            "Hard-Hit %": b_recent["hard_hit_rate"],
            "Avg EV": b_recent["ev"],
            "Pitcher HR/PA": p90["hr_rate"],
            "Expected PA": result["expected_pa"],
            "Lineup": "Confirmed" if c["confirmed"] else "Lineup pending",
            "Batting Order": c["order"] if pd.notna(c["order"]) else np.nan,
            "Reason": "",
            "Lineup Penalty": result["lineup_penalty"],
            "Data Confidence": result["data_confidence"],
            "Pick Type": "",
        })

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        return result_df

    result_df = result_df.sort_values(
        ["HR Probability", "Score", "Data Confidence"],
        ascending=False,
    ).reset_index(drop=True)

    result_df["Reason"] = result_df.apply(reason, axis=1)
    result_df["Pick Type"] = [
        pick_label(row, i + 1, 0)
        for i, (_, row) in enumerate(result_df.iterrows())
    ]
    return result_df


if run:
    cache_key = f"{selected_date.isoformat()}_{recent_days}"
    st.session_state["prediction_key"] = cache_key

    if (
        st.session_state.get("cached_prediction_key") == cache_key
        and isinstance(st.session_state.get("cached_predictions"), pd.DataFrame)
        and not st.session_state["cached_predictions"].empty
    ):
        result_df = st.session_state["cached_predictions"].copy()
    else:
        with st.spinner(
            "⚡ Building today's predictions… first run may take longer; "
            "repeat runs use the cached result."
        ):
            result_df = build_daily_predictions(
                selected_date.isoformat(),
                recent_days,
            )
        st.session_state["cached_prediction_key"] = cache_key
        st.session_state["cached_predictions"] = result_df.copy()

    if result_df.empty:
        st.warning("No MLB games or hitters could be evaluated for this date.")
        st.stop()

    for rank, (_, h) in enumerate(result_df.head(10).iterrows(), start=1):
        st.session_state.setdefault("prediction_history", []).append({
            "date": selected_date.isoformat(),
            "game_pk": int(h["Game PK"]),
            "player_id": int(h["Player ID"]),
            "player": h["Player"],
            "rank": rank,
            "probability": float(h["HR Probability"]),
        })

    # -------------------------
    # Daily board
    # -------------------------
    best = result_df.iloc[0]
    st.subheader("🏆 Today's HR Board")
    st.caption("The model explicitly discounts hitters whose starting lineup has not been confirmed.")

    a, b, c, d = st.columns(4)
    with a: st.metric("🥇 Best Pick", best["Player"])
    with b: st.metric("HR Probability", f'{best["HR Probability"]:.1%}')
    with c: st.metric("Model Score", f'{best["Score"]:.0f}/100')
    with d: st.metric("Confidence", best["Confidence"])

    st.success(f'**{best["Player"]}** — {best["Team"]} vs {best["Opponent"]}. {best["Reason"].capitalize()}.')

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

    st.subheader("🎯 Pick Types")
    confirmed_top = result_df[result_df["Lineup"] == "Confirmed"].head(10)
    if not confirmed_top.empty:
        value = confirmed_top.sort_values(["Score", "HR Probability"], ascending=False).iloc[0]
        sleeper_candidates = confirmed_top[confirmed_top["Barrel %"] >= 0.08]
        sleeper = sleeper_candidates.sort_values("HR Probability", ascending=False).iloc[0] if not sleeper_candidates.empty else value
        x, y = st.columns(2)
        with x: st.info(f"💎 **Value Pick:** {value['Player']} — {value['HR Probability']:.1%} HR probability, {value['Score']:.0f}/100 score.")
        with y: st.info(f"🌙 **Sleeper Pick:** {sleeper['Player']} — {sleeper['HR Probability']:.1%} HR probability, {sleeper['Barrel %']:.1%} barrels.")

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
    columns = ["Player", "Team", "Opponent", "Opposing SP", "HR Probability", "Score", "Confidence", "Pick Type", "Lineup", "Barrel %", "Hard-Hit %", "Avg EV", "Pitcher HR/PA", "Matchup HR/PA", "Expected PA", "Weather", "Temp F", "Wind MPH", "Venue"]
    st.dataframe(display[columns], use_container_width=True, hide_index=True)

    st.subheader("🔎 Why the top picks rank highly")
    for i, row in result_df.head(min(5, top_n)).iterrows():
        st.write(f"**#{i+1} {row['Player']} — {row['HR Probability']:.1%}**  \n{row['Reason'].capitalize()}.")

    st.caption("Best practice: run the model again after official batting orders are posted. The model treats lineup confirmation and batting-order position as opportunity information.")

    st.subheader("📈 Model Performance")
    hist = load_prediction_history()
    if hist is not None and not hist.empty:
        checked = hist[hist["actual_hr"].notna()].copy()
        if not checked.empty:
            top5h = checked[checked["rank"] <= 5]
            top1h = checked[checked["rank"] == 1]
            m1, m2, m3 = st.columns(3)
            with m1: st.metric("Top Pick HR Rate", f'{top1h["actual_hr"].gt(0).mean():.1%}' if not top1h.empty else "—")
            with m2: st.metric("Top 5 HR Rate", f'{top5h["actual_hr"].gt(0).mean():.1%}' if not top5h.empty else "—")
            with m3: st.metric("Predictions Checked", f'{len(checked)}')
            st.caption("These are simple hit rates from predictions retained in this browser session. They are not a full historical backtest.")
            st.dataframe(checked[["date","rank","player","probability","actual_hr"]].sort_values(["date","rank"], ascending=[False, True]).head(50), use_container_width=True, hide_index=True)
        else:
            st.info("Predictions are saved for this session. Reopen the app after games finish and use the same session to check results.")
    else:
        st.info("Run predictions to begin building a performance history.")

    st.download_button("⬇️ Download today's rankings as CSV", result_df.to_csv(index=False).encode("utf-8"), file_name=f"mlb_hr_predictions_{selected_date.isoformat()}.csv", mime="text/csv")

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

st.caption("MLB HR Predictor v11 • Fast daily cache + parallel Statcast layer • Data: MLB StatsAPI + Baseball Savant/Statcast")
