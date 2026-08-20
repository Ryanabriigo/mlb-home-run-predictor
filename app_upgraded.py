
import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import date, timedelta
from pybaseball import statcast_batter, statcast_pitcher

# ============================================================
# MLB DAILY HOME RUN PREDICTOR — v2
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
@st.cache_data(ttl=1800)
def get_batter_window(player_id, start_date, end_date):
    try:
        df = statcast_batter(start_date, end_date, player_id)
    except Exception:
        return {
            "pa": 0,
            "hr": 0,
            "hr_pa": LEAGUE_HR_PA,
            "barrel_rate": DEFAULT_BARREL,
            "hard_hit_rate": DEFAULT_HARD_HIT,
            "ev": DEFAULT_EV,
            "bbe": 0,
        }

    if df is None or df.empty:
        return {
            "pa": 0,
            "hr": 0,
            "hr_pa": LEAGUE_HR_PA,
            "barrel_rate": DEFAULT_BARREL,
            "hard_hit_rate": DEFAULT_HARD_HIT,
            "ev": DEFAULT_EV,
            "bbe": 0,
        }

    pa_df = df[df["events"].notna()].copy()
    bbe = df[df["launch_speed"].notna()].copy()

    pa = len(pa_df)
    hr = int((pa_df["events"] == "home_run").sum())

    if len(bbe):
        hard_hit = float((bbe["launch_speed"] >= 95).mean())
        ev = float(bbe["launch_speed"].mean())
    else:
        hard_hit = DEFAULT_HARD_HIT
        ev = DEFAULT_EV

    # Statcast's launch_speed_angle code 6 represents a barrel in
    # pybaseball's returned data.
    if len(bbe) and "launch_speed_angle" in bbe.columns:
        barrel_rate = float((bbe["launch_speed_angle"] == 6).mean())
    else:
        barrel_rate = DEFAULT_BARREL

    # Shrink small samples toward league-average priors.
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


@st.cache_data(ttl=1800)
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


# -------------------------
# Model
# -------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def model_player(batter_30, batter_90, pitcher_90, confirmed, lineup_slot):
    # Recent form gets meaningful weight, but 90-day performance stabilizes
    # the estimate when the recent sample is small.
    hr_signal = (
        0.60 * (batter_30["hr_pa"] / LEAGUE_HR_PA)
        + 0.40 * (batter_90["hr_pa"] / LEAGUE_HR_PA)
    )

    barrel_signal = batter_30["barrel_rate"] / DEFAULT_BARREL
    hard_hit_signal = batter_30["hard_hit_rate"] / DEFAULT_HARD_HIT
    ev_signal = batter_30["ev"] / DEFAULT_EV

    pitcher_signal = pitcher_90["hr_rate"] / LEAGUE_HR_PA

    # Multiplicative quality index, capped to avoid tiny samples exploding.
    hitter_quality = (
        0.45 * hr_signal
        + 0.25 * barrel_signal
        + 0.20 * hard_hit_signal
        + 0.10 * ev_signal
    )
    hitter_quality = clamp(hitter_quality, 0.35, 3.5)

    pitcher_quality = clamp(0.65 + 0.35 * pitcher_signal, 0.55, 2.25)

    # Top-of-order hitters generally receive more PA opportunities.
    pa_by_slot = {
        1: 4.55, 2: 4.45, 3: 4.35, 4: 4.25, 5: 4.10,
        6: 3.95, 7: 3.80, 8: 3.65, 9: 3.55
    }
    expected_pa = pa_by_slot.get(int(lineup_slot) if pd.notna(lineup_slot) else 5, 3.75)

    # If the lineup isn't confirmed, reduce opportunity confidence rather
    # than pretending the player has a known batting-order slot.
    if not confirmed:
        expected_pa = min(expected_pa, 3.75)

    # Baseline HR/PA adjusted by hitter quality and pitcher HR tendency.
    adjusted_hr_pa = LEAGUE_HR_PA * hitter_quality * pitcher_quality
    adjusted_hr_pa = clamp(adjusted_hr_pa, 0.005, 0.12)

    # P(at least one HR) over expected PA.
    probability = 1.0 - (1.0 - adjusted_hr_pa) ** expected_pa

    # Confidence is separate from probability: small samples and unconfirmed
    # lineups should lower confidence even when the raw score is attractive.
    sample_score = clamp(
        (batter_30["pa"] / 80.0) * 50 + (batter_90["pa"] / 220.0) * 50,
        0, 100
    )
    lineup_score = 100 if confirmed else 45
    data_confidence = 0.65 * sample_score + 0.35 * lineup_score

    if data_confidence >= 80:
        confidence = "HIGH"
    elif data_confidence >= 60:
        confidence = "GOOD"
    else:
        confidence = "WATCH"

    score = 100 * clamp(
        0.38 * min(hitter_quality / 2.5, 1)
        + 0.22 * min(pitcher_quality / 1.6, 1)
        + 0.18 * min(barrel_signal / 2.0, 1)
        + 0.12 * min(hard_hit_signal / 1.5, 1)
        + 0.10 * min(expected_pa / 4.5, 1),
        0, 1
    )

    return {
        "probability": float(probability),
        "score": float(score),
        "expected_pa": float(expected_pa),
        "confidence": confidence,
        "data_confidence": float(data_confidence),
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

if run:
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

            p90 = get_pitcher_window(
                pitcher_id,
                start_90.isoformat(),
                selected_date.isoformat(),
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

            for _, hitter in hitters.iterrows():
                pid = int(hitter["id"])

                b30 = get_batter_window(
                    pid,
                    start_30.isoformat(),
                    selected_date.isoformat(),
                )
                b90 = get_batter_window(
                    pid,
                    start_90.isoformat(),
                    selected_date.isoformat(),
                )

                result = model_player(
                    b30,
                    b90,
                    p90,
                    confirmed,
                    hitter["order"] if pd.notna(hitter["order"]) else 5,
                )

                rows.append(
                    {
                        "Player": hitter["name"],
                        "Team": team,
                        "Opponent": opponent,
                        "Opposing SP": pitcher_name,
                        "Venue": game["venue"],
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
                    }
                )

            completed += 1
            progress.progress(completed / total_steps)

    result_df = pd.DataFrame(rows)

    if result_df.empty:
        st.warning("No hitters could be evaluated.")
        st.stop()

    result_df = result_df.sort_values(
        ["HR Probability", "Score"],
        ascending=False,
    ).reset_index(drop=True)

    result_df["Reason"] = result_df.apply(reason, axis=1)

    # -------------------------
    # Best pick
    # -------------------------
    best = result_df.iloc[0]

    st.subheader("🥇 Best HR Pick")

    a, b, c, d = st.columns(4)
    with a:
        st.metric("Player", best["Player"])
    with b:
        st.metric("HR Probability", f'{best["HR Probability"]:.1%}')
    with c:
        st.metric("Model Score", f'{best["Score"]:.0f}/100')
    with d:
        st.metric("Confidence", best["Confidence"])

    st.info(
        f'**{best["Player"]}** — {best["Team"]} vs {best["Opponent"]}. '
        f'{best["Reason"].capitalize()}.'
    )

    # -------------------------
    # Top picks table
    # -------------------------
    st.subheader(f"🔥 Top {top_n} HR Candidates")

    display = result_df.head(top_n).copy()

    display["HR Probability"] = display["HR Probability"].map(lambda x: f"{x:.1%}")
    display["Score"] = display["Score"].map(lambda x: f"{x:.0f}")
    display["HR/PA"] = display["HR/PA"].map(lambda x: f"{x:.3f}")
    display["Barrel %"] = display["Barrel %"].map(lambda x: f"{x:.1%}")
    display["Hard-Hit %"] = display["Hard-Hit %"].map(lambda x: f"{x:.1%}")
    display["Pitcher HR/PA"] = display["Pitcher HR/PA"].map(lambda x: f"{x:.3f}")
    display["Avg EV"] = display["Avg EV"].map(lambda x: f"{x:.1f}")
    display["Expected PA"] = display["Expected PA"].map(lambda x: f"{x:.1f}")

    columns = [
        "Player", "Team", "Opponent", "Opposing SP",
        "HR Probability", "Score", "Confidence",
        "Lineup", "Barrel %", "Hard-Hit %", "Avg EV",
        "Pitcher HR/PA", "Expected PA", "Venue",
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
        "Small samples are shrunk toward league-average priors."
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

st.caption("MLB HR Predictor v2 • Data: MLB StatsAPI + Baseball Savant/Statcast via pybaseball")
