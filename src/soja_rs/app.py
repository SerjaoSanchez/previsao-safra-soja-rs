"""App Streamlit de entrega (Fase 4).

Mapa coroplético do RS com previsão por município, seletor de modelo,
ano-safra e busca por município (com destaque no mapa e histórico
previsto vs. observado), painel previsto vs. observado e evolução do erro
por ano.

Lê de ``data/processed/app_data.duckdb``, um banco enxuto (só as tabelas
``validacao_previsoes``/``validacao_metricas``/``municipio_nomes``) gerado
por ``exportar_app_data()`` em ``soja_rs.train`` — versionado no git à
parte do ``soja_rs.duckdb`` principal, que tem a série de clima diário
(~8M linhas) e é grande demais para o repositório.

Rode com ``make app`` ou ``streamlit run src/soja_rs/app.py``. Rode
``make train`` antes se ``data/processed/app_data.duckdb`` ainda não
existir.
"""

import json

import duckdb
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from soja_rs.data import APP_DB_PATH, RAW_DIR

st.set_page_config(page_title="Previsão de safra de soja — RS", layout="wide")

GEOJSON_PATH = RAW_DIR / "malha" / "rs_43_municipios.geojson"
TODOS_OS_MUNICIPIOS = "Todos os municípios"
COR_DESTAQUE = "#e63946"


@st.cache_data
def carregar_dados():
    if not APP_DB_PATH.exists():
        return None, None, None, None
    with duckdb.connect(str(APP_DB_PATH), read_only=True) as con:
        previsoes = con.execute("SELECT * FROM validacao_previsoes").df()
        metricas = con.execute("SELECT * FROM validacao_metricas").df()
        nomes = con.execute("SELECT * FROM municipio_nomes").df()
    with open(GEOJSON_PATH) as f:
        malha = json.load(f)
    return previsoes, metricas, malha, nomes


def _destacar_municipio(fig, malha, municipio_id):
    """Sobrepõe uma borda colorida no polígono do município escolhido."""
    fig.add_trace(
        go.Choropleth(
            geojson=malha,
            locations=[municipio_id],
            z=[1],
            featureidkey="properties.codarea",
            showscale=False,
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
            marker_line_color=COR_DESTAQUE,
            marker_line_width=3.5,
            hoverinfo="skip",
        )
    )


def main():
    previsoes, metricas, malha, nomes = carregar_dados()

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

    previsoes = previsoes.merge(nomes, on="municipio_id", how="left")

    modelos = sorted(previsoes["modelo"].unique())
    modelo_default = modelos.index("lightgbm") if "lightgbm" in modelos else 0
    modelo = st.sidebar.selectbox("Modelo", modelos, index=modelo_default)

    anos = sorted(previsoes["ano"].unique())
    ano = st.sidebar.selectbox("Ano-safra", anos, index=len(anos) - 1)

    st.sidebar.divider()
    nomes_ordenados = sorted(nomes["municipio_nome"].dropna().unique())
    municipio_nome = st.sidebar.selectbox(
        "Buscar município",
        [TODOS_OS_MUNICIPIOS] + nomes_ordenados,
        help="Clique e comece a digitar para filtrar a lista.",
    )

    dados_modelo = previsoes[previsoes["modelo"] == modelo]
    sub = dados_modelo[dados_modelo["ano"] == ano]
    metrica_ano = metricas[(metricas["modelo"] == modelo) & (metricas["ano"] == ano)]

    if not metrica_ano.empty:
        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE do ano", f"{metrica_ano['rmse'].iloc[0]:.0f} kg/ha")
        col2.metric("R² dentro do ano", f"{metrica_ano['r2_dentro_do_ano'].iloc[0]:.2f}")
        col3.metric("Municípios testados", int(metrica_ano["n"].iloc[0]))

    municipio_selecionado = municipio_nome != TODOS_OS_MUNICIPIOS
    if municipio_selecionado:
        municipio_id_sel = nomes.loc[
            nomes["municipio_nome"] == municipio_nome, "municipio_id"
        ].iloc[0]

    col_mapa, col_dispersao = st.columns([3, 2])

    with col_mapa:
        # px.choropleth (SVG/D3), não choropleth_mapbox: não depende de
        # WebGL nem de tiles externos, mais robusto e mais alinhado com a
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
            hover_name="municipio_nome",
            hover_data={"municipio_id": False, "y_true": ":.0f", "y_pred": ":.0f"},
            labels={"y_pred": "Previsto (kg/ha)", "y_true": "Observado (kg/ha)"},
        )
        fig_mapa.update_geos(visible=False)
        if municipio_selecionado:
            _destacar_municipio(fig_mapa, malha, municipio_id_sel)
        fig_mapa.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0}, height=550)
        st.plotly_chart(fig_mapa, use_container_width=True)

    with col_dispersao:
        fig_disp = px.scatter(
            sub,
            x="y_true",
            y="y_pred",
            hover_name="municipio_nome",
            labels={"y_true": "Observado (kg/ha)", "y_pred": "Previsto (kg/ha)"},
        )
        if not sub.empty:
            limite = max(sub["y_true"].max(), sub["y_pred"].max()) * 1.05
            linha = {"dash": "dash", "color": "gray"}
            fig_disp.add_shape(type="line", x0=0, y0=0, x1=limite, y1=limite, line=linha)
        if municipio_selecionado:
            destaque = sub[sub["municipio_id"] == municipio_id_sel]
            fig_disp.add_scatter(
                x=destaque["y_true"],
                y=destaque["y_pred"],
                mode="markers",
                marker={"color": COR_DESTAQUE, "size": 16, "symbol": "star"},
                name=municipio_nome,
                showlegend=False,
            )
        fig_disp.update_layout(height=550, title="Previsto vs. observado")
        st.plotly_chart(fig_disp, use_container_width=True)

    st.subheader("Evolução do erro por ano")
    resumo_anos = metricas[metricas["modelo"] == modelo].sort_values("ano")
    fig_rmse = px.bar(
        resumo_anos, x="ano", y="rmse", labels={"rmse": "RMSE (kg/ha)", "ano": "Ano-safra"}
    )
    st.plotly_chart(fig_rmse, use_container_width=True)

    if municipio_selecionado:
        st.subheader(f"Perfil: {municipio_nome}")
        historico = dados_modelo[dados_modelo["municipio_id"] == municipio_id_sel]
        historico = historico.sort_values("ano")

        if historico.empty:
            st.info(
                "Sem histórico de validação para este município (área colhida "
                "abaixo do filtro de qualidade, ou anos insuficientes)."
            )
        else:
            erro_medio = (historico["y_true"] - historico["y_pred"]).abs().mean()
            vies_medio = (historico["y_pred"] - historico["y_true"]).mean()
            sinal = "superestima" if vies_medio > 0 else "subestima"

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Erro absoluto médio", f"{erro_medio:.0f} kg/ha")
            col_b.metric(
                "Viés médio",
                f"{vies_medio:+.0f} kg/ha",
                help=f"Em média, o modelo {sinal} este município.",
            )
            col_c.metric("Anos testados", len(historico))

            fig_hist = go.Figure()
            fig_hist.add_scatter(
                x=historico["ano"], y=historico["y_true"], name="Observado", mode="lines+markers"
            )
            fig_hist.add_scatter(
                x=historico["ano"], y=historico["y_pred"], name="Previsto", mode="lines+markers"
            )
            fig_hist.update_layout(
                height=400,
                xaxis_title="Ano-safra",
                yaxis_title="Rendimento (kg/ha)",
                legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
            )
            st.plotly_chart(fig_hist, use_container_width=True)


main()
