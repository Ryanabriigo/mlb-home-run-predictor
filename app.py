import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import date, timedelta
from pybaseball import statcast_batter, statcast_pitcher

API="https://statsapi.mlb.com/api/v1"
st.set_page_config(page_title="MLB HR Predictor",page_icon="⚾",layout="wide")

@st.cache_data(ttl=300)
def get(path,params=None):
    r=requests.get(f"{API}/{path}",params=params,timeout=30)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=300)
def games(day):
    x=get("schedule",{"sportId":1,"date":day,"hydrate":"probablePitcher,team,venue"})
    out=[]
    for d in x.get("dates",[]):
        for g in d.get("games",[]):
            out.append({
                "pk":g["gamePk"],"away":g["teams"]["away"]["team"]["abbreviation"],
                "home":g["teams"]["home"]["team"]["abbreviation"],
                "away_id":g["teams"]["away"]["team"]["id"],"home_id":g["teams"]["home"]["team"]["id"],
                "away_sp":g["teams"]["away"].get("probablePitcher",{}),
                "home_sp":g["teams"]["home"].get("probablePitcher",{}),
                "venue":g.get("venue",{}).get("name","")})
    return pd.DataFrame(out)

@st.cache_data(ttl=1800)
def roster(team_id):
    x=get(f"teams/{team_id}/roster",{"season":date.today().year})
    rows=[]
    for p in x.get("roster",[]):
        pos=p.get("position",{}).get("abbreviation","")
        if pos not in ("P","TWP"):
            rows.append((p["person"]["id"],p["person"]["fullName"]))
    return rows

@st.cache_data(ttl=1800)
def batter(pid,start,end):
    try: df=statcast_batter(start,end,pid)
    except Exception: return (.035,.07,.37,88.5)
    if df is None or df.empty: return (.035,.07,.37,88.5)
    ev=df[df["events"].notna()]; bbe=df[df["launch_speed"].notna()]
    pa=len(ev); hr=(ev["events"]=="home_run").sum()
    barrel=(bbe["launch_speed_angle"]==6).mean() if len(bbe) and "launch_speed_angle" in bbe else .07
    hard=(bbe["launch_speed"]>=95).mean() if len(bbe) else .37
    velo=bbe["launch_speed"].mean() if len(bbe) else 88.5
    return (hr/max(pa,1),barrel,hard,velo)

@st.cache_data(ttl=1800)
def pitcher(pid,start,end):
    if not pid: return .035
    try: df=statcast_pitcher(start,end,pid)
    except Exception: return .035
    if df is None or df.empty: return .035
    ev=df[df["events"].notna()]
    return (ev["events"]=="home_run").mean() if len(ev) else .035

@st.cache_data(ttl=300)
def lineup_ids(pk,side):
    try:
        x=get(f"game/{pk}/feed/live")
        players=x.get("liveData",{}).get("boxscore",{}).get("teams",{}).get(side,{}).get("players",{})
        ids=[]
        for p in players.values():
            bo=p.get("battingOrder")
            if bo and str(bo).endswith("00"): ids.append(p["person"]["id"])
        return ids
    except Exception: return []

def probability(hr,barrel,hard,velo,phr,confirmed):
    league=.035
    quality=.50*(hr/league)+.25*(barrel/.07)+.15*(hard/.37)+.10*(velo/88.5)
    matchup=.60*(phr/league)+.40
    pa=4.3 if confirmed else 3.7
    ppa=np.clip(league*quality*matchup,.003,.16)
    p=1-(1-ppa)**pa
    score=np.clip(50*quality+25*matchup,0,100)
    return float(p),float(score)

st.title("⚾ MLB Daily Home Run Predictor")
st.caption("Statcast-based estimates. Not guarantees or betting advice.")

day=st.date_input("Game date",date.today())
lookback=st.slider("Statcast lookback (days)",7,60,30)
top=st.slider("Show top candidates",5,30,15)

if st.button("🚀 RUN TODAY'S PREDICTIONS",type="primary",use_container_width=True):
    gs=games(day.isoformat())
    if gs.empty:
        st.warning("No MLB games found.")
        st.stop()
    start=(day-timedelta(days=lookback)).isoformat()
    rows=[]
    progress=st.progress(0)
    total=len(gs)*2; n=0
    for _,g in gs.iterrows():
        for side in ("away","home"):
            team=g[side]; tid=int(g[f"{side}_id"])
            sp=g["home_sp"] if side=="away" else g["away_sp"]
            spid=sp.get("id"); spname=sp.get("fullName","TBD")
            phr=pitcher(spid,start,day.isoformat())
            confirmed_ids=lineup_ids(int(g.pk),side)
            confirmed=bool(confirmed_ids)
            r=roster(tid)
            if confirmed: r=[x for x in r if x[0] in confirmed_ids]
            for pid,name in r:
                hr,barrel,hard,velo=batter(pid,start,day.isoformat())
                p,s=probability(hr,barrel,hard,velo,phr,confirmed)
                rows.append([name,team,g["home"] if side=="away" else g["away"],spname,g.venue,p,s,hr,barrel,hard,velo,"Confirmed" if confirmed else "Lineup pending"])
            n+=1; progress.progress(n/total)
    df=pd.DataFrame(rows,columns=["Player","Team","Opponent","Opposing SP","Venue","HR Probability","Score","HR/PA","Barrel","Hard-Hit","Avg EV","Lineup"])
    df=df.sort_values(["HR Probability","Score"],ascending=False).head(top)
    for c in ["HR Probability","Barrel","Hard-Hit"]:
        df[c]=df[c].map(lambda x:f"{x:.1%}")
    df["Score"]=df["Score"].map(lambda x:f"{x:.0f}")
    df["HR/PA"]=df["HR/PA"].map(lambda x:f"{x:.3f}")
    df["Avg EV"]=df["Avg EV"].map(lambda x:f"{x:.1f}")
    st.subheader(f"Top {top} — {day:%B %d, %Y}")
    st.dataframe(df,use_container_width=True,hide_index=True)
    st.info("Run again after official lineups are posted for the strongest version of the model.")

st.divider()
st.write("Inputs: recent HR rate, barrels, hard-hit rate, exit velocity, opposing starter HR tendency, expected plate appearances, and confirmed lineup status.")
