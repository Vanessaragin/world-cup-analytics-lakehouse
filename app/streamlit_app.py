from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GOLD_DIR = PROJECT_ROOT / "data" / "gold" / "openfootball"
SILVER_DIR = PROJECT_ROOT / "data" / "silver" / "openfootball"


st.set_page_config(
    page_title="World Cup Analytics",
    layout="wide"
)


@st.cache_data
def load_data():
    team_ranking = pd.read_csv(GOLD_DIR / "gold_team_ranking.csv")
    world_cups = pd.read_csv(GOLD_DIR / "gold_world_cup_summary.csv")
    matches = pd.read_csv(SILVER_DIR / "silver_matches.csv")

    return team_ranking, world_cups, matches


team_ranking, world_cups, matches = load_data()


st.title("World Cup Analytics Lakehouse")

st.markdown(
    "Dashboard analítico com dados históricos da Copa do Mundo."
)


total_cups = world_cups["year"].nunique()
total_games = world_cups["games"].sum()
total_goals = world_cups["goals"].sum()
avg_goals = total_goals / total_games


col1, col2, col3, col4 = st.columns(4)

col1.metric("Copas", int(total_cups))
col2.metric("Jogos", int(total_games))
col3.metric("Gols", int(total_goals))
col4.metric("Média de gols", round(avg_goals, 2))


st.subheader("Ranking histórico das seleções")

top_teams = team_ranking.head(10)

fig_teams = px.bar(
    top_teams,
    x="team",
    y="ranking_score",
    hover_data=["games", "wins", "goals_for", "goal_difference"],
    title="Top 10 seleções por ranking histórico"
)

st.plotly_chart(fig_teams, use_container_width=True)


st.subheader("Copas mais ofensivas")

fig_cups = px.bar(
    world_cups.sort_values("avg_goals_per_game", ascending=False).head(10),
    x="year",
    y="avg_goals_per_game",
    hover_data=["games", "goals", "max_goals_in_match"],
    title="Top 10 Copas por média de gols"
)

st.plotly_chart(fig_cups, use_container_width=True)


st.subheader("Análise por seleção")

selected_team = st.selectbox(
    "Escolha uma seleção",
    team_ranking["team"].sort_values()
)

team_data = team_ranking[
    team_ranking["team"] == selected_team
].iloc[0]

col1, col2, col3, col4 = st.columns(4)

col1.metric("Jogos", int(team_data["games"]))
col2.metric("Vitórias", int(team_data["wins"]))
col3.metric("Gols feitos", int(team_data["goals_for"]))
col4.metric("Saldo de gols", int(team_data["goal_difference"]))


team_matches = matches[
    (matches["team1_normalized"] == selected_team)
    | (matches["team2_normalized"] == selected_team)
]

st.dataframe(
    team_matches[
        [
            "year",
            "round",
            "date",
            "team1_normalized",
            "team2_normalized",
            "team1_score_final",
            "team2_score_final",
            "winner"
        ]
    ],
    use_container_width=True
)