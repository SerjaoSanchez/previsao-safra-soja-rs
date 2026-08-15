"""App Streamlit de entrega (Fase 4).

Mapa coroplético do RS com previsão por município, seletor de modelo e
ano-safra, painel previsto vs. observado e evolução do erro por ano.

Rode com ``make app`` ou ``streamlit run src/soja_rs/app.py``. Depende das
tabelas ``validacao_previsoes``/``validacao_metricas`` no DuckDB — rode
``make train`` antes se elas ainda não existirem.
"""

import json

import duckdb
import plotly.express as px
import streamlit as st

from soja_rs.data import DUCKDB_PATH, RAW_DIR

st.set_page_config(page_title="Previsão de safra de soja — RS", layout="wide")

GEOJSON_PATH = RAW_DIR / "malha" / "rs_43_municipios.geojson"


@st.cache_data
def carregar_dados():
    with duckdb.connect(str(DUCKDB_PATH), read_only=True) as con:
        tabelas = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        if "validacao_previsoes" not in tabelas:
            return None, None, None
        previsoes = con.execute("SELECT * FROM validacao_previsoes").df()
        metricas = con.execute("SELECT * FROM validacao_metricas").df()
    with open(GEOJSON_PATH) as f:
        malha = json.load(f)
    return previsoes, metricas, malha


def main():
    previsoes, metricas, malha = carregar_dados()

    st.title("Previsão de safra de soja no Rio Grande do Sul")
    st.caption(
        "Rendimento médio (kg/ha) por município, previsto com informação "
        "climática disponível até 31 de janeiro do próprio ano-safra."
    )

    if previsoes is None:
        st.error(
            "Ainda não há resultados de validação no banco. Rode `make data`, "
            "`make features` e `make train` (nessa ordem) antes de abrir o app."
        )
        st.stop()

    modelos = sorted(previsoes["modelo"].unique())
    modelo_default = modelos.index("lightgbm") if "lightgbm" in modelos else 0
    modelo = st.sidebar.selectbox("Modelo", modelos, index=modelo_default)

    anos = sorted(previsoes["ano"].unique())
    ano = st.sidebar.selectbox("Ano-safra", anos, index=len(anos) - 1)

    sub = previsoes[(previsoes["modelo"] == modelo) & (previsoes["ano"] == ano)]
    metrica_ano = metricas[(metricas["modelo"] == modelo) & (metricas["ano"] == ano)]

    if not metrica_ano.empty:
        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE do ano", f"{metrica_ano['rmse'].iloc[0]:.0f} kg/ha")
        col2.metric("R² dentro do ano", f"{metrica_ano['r2_dentro_do_ano'].iloc[0]:.2f}")
        col3.metric("Municípios testados", int(metrica_ano["n"].iloc[0]))

    col_mapa, col_dispersao = st.columns([3, 2])

    with col_mapa:
        # px.choropleth (SVG/D3), não choropleth_mapbox: não depende de
        # WebGL nem de tiles externos — mais robusto e mais alinhado com a
        # reprodutibilidade offline do resto do projeto.
        fig_mapa = px.choropleth(
            sub,
            geojson=malha,
            locations="municipio_id",
            featureidkey="properties.codarea",
            color="y_pred",
            color_continuous_scale="YlGn",
            scope="south america",
            fitbounds="locations",
            labels={"y_pred": "Previsto (kg/ha)"},
        )
        fig_mapa.update_geos(visible=False)
        fig_mapa.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0}, height=550)
        st.plotly_chart(fig_mapa, use_container_width=True)

    with col_dispersao:
        fig_disp = px.scatter(
            sub,
            x="y_true",
            y="y_pred",
            hover_data=["municipio_id"],
            labels={"y_true": "Observado (kg/ha)", "y_pred": "Previsto (kg/ha)"},
        )
        if not sub.empty:
            limite = max(sub["y_true"].max(), sub["y_pred"].max()) * 1.05
            linha = {"dash": "dash", "color": "gray"}
            fig_disp.add_shape(type="line", x0=0, y0=0, x1=limite, y1=limite, line=linha)
        fig_disp.update_layout(height=550, title="Previsto vs. observado")
        st.plotly_chart(fig_disp, use_container_width=True)

    st.subheader("Evolução do erro por ano")
    resumo_anos = metricas[metricas["modelo"] == modelo].sort_values("ano")
    fig_rmse = px.bar(
        resumo_anos, x="ano", y="rmse", labels={"rmse": "RMSE (kg/ha)", "ano": "Ano-safra"}
    )
    st.plotly_chart(fig_rmse, use_container_width=True)


main()
