import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="MLB Stat Comparison", layout="centered")

STATS_API = "https://statsapi.mlb.com/api/v1/stats"

# api_key -> (display label, lower_is_better, decimals)
# Playing time leads: it's the sample size every rate stat below rests on.
BATTING_STATS = {
    "plateAppearances": ("Plate Appearances", False, 0),
    "avg": ("Batting Average", False, 3),
    "obp": ("On-Base %", False, 3),
    "slg": ("Slugging %", False, 3),
    "ops": ("OPS", False, 3),
    "homeRuns": ("Home Runs", False, 0),
    "rbi": ("RBI", False, 0),
    "stolenBases": ("Stolen Bases", False, 0),
    "baseOnBalls": ("Walks", False, 0),
    "strikeOuts": ("Strikeouts", True, 0),
}

PITCHING_STATS = {
    "inningsPitched": ("Innings Pitched", False, 1),
    "era": ("ERA", True, 2),
    "whip": ("WHIP", True, 2),
    "strikeoutsPer9Inn": ("K per 9", False, 2),
    "walksPer9Inn": ("BB per 9", True, 2),
    "homeRunsPer9": ("HR per 9", True, 2),
    "strikeOuts": ("Strikeouts", False, 0),
    "wins": ("Wins", False, 0),
    "saves": ("Saves", False, 0),
}

# Minimum playing time per team game for a player to count toward the
# distribution. Not MLB's official qualifier — deliberately looser, so
# part-time hitters and relievers are still represented, while very small
# samples that would distort rate stats are excluded.
MIN_PA_PER_GAME = 1.0
MIN_IP_PER_GAME = 0.25


@st.cache_data(ttl=60 * 60 * 12)
def current_season() -> int:
    resp = requests.get(
        "https://statsapi.mlb.com/api/v1/seasons/current", params={"sportId": 1}, timeout=30
    )
    resp.raise_for_status()
    return int(resp.json()["seasons"][0]["seasonId"])


@st.cache_data(ttl=60 * 30)
def team_games_played(season: int) -> int:
    """Median games played across all 30 teams — scales the playing-time floor."""
    resp = requests.get(
        "https://statsapi.mlb.com/api/v1/standings",
        params={"leagueId": "103,104", "season": season, "standingsTypes": "regularSeason"},
        timeout=30,
    )
    resp.raise_for_status()
    games = [
        team["gamesPlayed"]
        for record in resp.json()["records"]
        for team in record["teamRecords"]
    ]
    return int(pd.Series(games).median()) if games else 162


def _fetch(group: str, season: int) -> pd.DataFrame:
    resp = requests.get(
        STATS_API,
        params={
            "stats": "season",
            "group": group,
            "season": season,
            "sportId": 1,
            "gameType": "R",
            "playerPool": "All",
            "limit": 3000,
        },
        timeout=60,
    )
    resp.raise_for_status()

    rows = []
    for split in resp.json()["stats"][0]["splits"]:
        row = {
            "playerId": split["player"]["id"],
            "Name": split["player"]["fullName"],
            "Team": split.get("team", {}).get("name", ""),
            "Pos": split.get("position", {}).get("abbreviation", ""),
        }
        row.update(split["stat"])
        rows.append(row)

    df = pd.DataFrame(rows)
    numeric = set(BATTING_STATS) | set(PITCHING_STATS) | {"plateAppearances", "inningsPitched"}
    for col in numeric & set(df.columns):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.drop_duplicates(subset="playerId", keep="first").set_index("playerId")


@st.cache_data(show_spinner="Loading league data...", ttl=60 * 30)
def load_season(season: int):
    hitting = _fetch("hitting", season)
    pitching = _fetch("pitching", season)

    # A player's listed position in the pitching feed is the reliable signal:
    # a shortstop who mopped up an inning is still listed SS, while a real
    # pitcher who took some at-bats is listed P in the hitting feed.
    position = {}
    position.update(hitting["Pos"].to_dict())
    position.update(pitching["Pos"].to_dict())

    def role_of(pid: int) -> str:
        pos = position.get(pid, "")
        if pos == "TWP":
            return "Two-Way"
        return "Pitching" if pos == "P" else "Batting"

    roles = {pid: role_of(pid) for pid in set(hitting.index) | set(pitching.index)}

    roster = pd.DataFrame(
        [
            {
                "playerId": pid,
                "Name": (pitching if pid in pitching.index else hitting).loc[pid, "Name"],
                "Team": (pitching if roles[pid] != "Batting" and pid in pitching.index else hitting).loc[pid, "Team"],
                "Role": roles[pid],
            }
            for pid in roles
        ]
    ).set_index("playerId")

    return hitting, pitching, roster


def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def percentile_for(series: pd.Series, value: float, lower_is_better: bool) -> float:
    """Share of the league at or below `value`, splitting credit for ties.

    Equivalent to scipy's percentileofscore(kind="mean"), inlined so the app
    doesn't carry scipy just for this one call.
    """
    values = series.dropna().to_numpy()
    if values.size == 0:
        return float("nan")
    below = (values < value).sum()
    at_or_below = (values <= value).sum()
    pct = (below + at_or_below) / (2 * values.size) * 100
    return 100 - pct if lower_is_better else pct


def plot_distribution(series: pd.Series, value: float, label: str, lower_is_better: bool):
    fig, ax = plt.subplots(figsize=(5, 2.2))
    ax.hist(series.dropna(), bins=25, color="#4C72B0", alpha=0.75)
    ax.axvline(value, color="#C44E52", linewidth=2)
    direction = "lower is better" if lower_is_better else "higher is better"
    ax.set_title(f"{label} — {direction}", fontsize=9)
    ax.set_yticks([])
    fig.tight_layout()
    return fig


def show_comparison(
    player_row: pd.Series,
    pool: pd.DataFrame,
    everyone: pd.DataFrame,
    stat_map: dict,
    heading: str,
    pt_col: str,
    pt_label: str,
    floor: int,
):
    available = {k: v for k, v in stat_map.items() if k in pool.columns}
    st.subheader(heading)
    st.caption(
        f"Stats compared against {len(pool)} players with {floor}+ {pt_label}; "
        f"playing time against all {len(everyone)}."
    )

    played = float(player_row[pt_col])
    if played < floor:
        st.warning(
            f"Small sample — {played:g} {pt_label}, under the {floor} {pt_label} needed to join "
            f"the comparison group, so the rate stats below can swing a lot on a few games."
        )

    summary = []
    cols = st.columns(2)
    for i, (key, (label, lower_is_better, decimals)) in enumerate(available.items()):
        value = float(player_row[key])
        # Playing time is what defines `pool`, so ranking it there would be
        # circular — measure it against every player instead.
        series = (everyone if key == pt_col else pool)[key]
        pct = percentile_for(series, value, lower_is_better)
        summary.append({
            "Stat": label,
            "Value": round(value, decimals),
            "League Median": round(series.median(), decimals),
            "Percentile": round(pct, 1),
        })
        with cols[i % 2]:
            st.metric(label, f"{value:.{decimals}f}", f"{ordinal(round(pct))} percentile")
            fig = plot_distribution(series, value, label, lower_is_better)
            st.pyplot(fig)
            plt.close(fig)

    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)


st.title("MLB Stat Comparison")

try:
    season = current_season()
    hitting, pitching, roster = load_season(season)
    games = team_games_played(season)
except Exception as e:
    st.error(f"Couldn't load MLB data: {e}")
    st.stop()

st.caption(
    f"{season} regular season · {games} team games played · updated live from the MLB Stats API"
)

min_pa = int(MIN_PA_PER_GAME * games)
min_ip = int(MIN_IP_PER_GAME * games)

# Everyone who filled that role at all, then the subset with enough playing
# time to give the rate stats a stable distribution.
all_batters = hitting[hitting.index.map(lambda p: roster.loc[p, "Role"]) != "Pitching"]
all_pitchers = pitching[pitching.index.map(lambda p: roster.loc[p, "Role"]) != "Batting"]

batting_pool = all_batters[all_batters["plateAppearances"] >= min_pa]
pitching_pool = all_pitchers[all_pitchers["inningsPitched"] >= min_ip]

chosen = st.selectbox(
    "Player",
    roster.sort_values("Name").index.tolist(),
    index=None,
    placeholder="Start typing a player's name...",
    format_func=lambda p: f"{roster.loc[p, 'Name']} ({roster.loc[p, 'Team']})",
)

if chosen is None:
    st.info("Pick any player to see how their season compares to the rest of the league.")
    st.stop()

role = roster.loc[chosen, "Role"]

if role in ("Batting", "Two-Way") and chosen in hitting.index:
    show_comparison(
        hitting.loc[chosen],
        batting_pool,
        all_batters,
        BATTING_STATS,
        "Batting" if role == "Two-Way" else "Batting vs. league",
        "plateAppearances",
        "PA",
        min_pa,
    )

if role in ("Pitching", "Two-Way") and chosen in pitching.index:
    show_comparison(
        pitching.loc[chosen],
        pitching_pool,
        all_pitchers,
        PITCHING_STATS,
        "Pitching" if role == "Two-Way" else "Pitching vs. league",
        "inningsPitched",
        "IP",
        min_ip,
    )
