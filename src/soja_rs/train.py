"""Modelagem e validação temporal (Fase 3).

Baselines obrigatórios, nesta ordem: média histórica do município → média +
tendência linear temporal → Ridge → LightGBM. Validação sempre temporal
expansiva (treina até o ano X, testa no ano X+1) — nunca K-fold aleatório,
que vazaria informação do mesmo ano entre treino e teste.

Rode com ``make train`` ou ``python -m soja_rs.train``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGET_COL = "rendimento_medio_kg_ha"
FEATURE_COLS = [
    "gdd_semeadura",
    "gdd_vegetativo",
    "gdd_reprodutivo",
    "precip_mm_semeadura",
    "precip_mm_vegetativo",
    "precip_mm_reprodutivo",
    "anomalia_precip_reprodutivo_mm",
    "maior_seca_reprodutivo_dias",
    "dias_tmax32_reprodutivo",
    "radiacao_mj_m2_reprodutivo",
    "balanco_hidrico_reprodutivo_mm",
    "oni_lag_son",
    "rendimento_medio_historico_kg_ha",
    "ano",
]
ANO_MINIMO_TREINO = 5  # anos de histórico exigidos antes do 1º ano testado


def montar_dataset(
    features: pd.DataFrame, pam: pd.DataFrame, area_minima_ha: float = 500.0
) -> pd.DataFrame:
    """Junta as features com o alvo (rendimento observado).

    Descarta município-anos sem histórico suficiente (1º ano de cada
    município, sem ``rendimento_medio_historico_kg_ha``) e os de área
    colhida pequena, cujo rendimento é ruidoso (ver base.txt, Fase 1).
    """
    alvo = pam[["municipio_id", "ano", TARGET_COL, "area_colhida_ha"]]
    df = features.merge(alvo, on=["municipio_id", "ano"], how="inner")
    df = df.dropna(subset=[TARGET_COL, "rendimento_medio_historico_kg_ha"])
    df = df[df["area_colhida_ha"] >= area_minima_ha]
    return df.reset_index(drop=True)


def _preparar_x(
    df: pd.DataFrame, referencia: pd.DataFrame, feature_cols: list[str]
) -> pd.DataFrame:
    """Preenche ausentes com a mediana do treino (nunca do próprio teste)."""
    medianas = referencia[feature_cols].median()
    return df[feature_cols].fillna(medianas)


def baseline_media_historica(df_treino: pd.DataFrame, df_teste: pd.DataFrame) -> np.ndarray:
    return df_teste["rendimento_medio_historico_kg_ha"].to_numpy()


def baseline_media_mais_tendencia(df_treino: pd.DataFrame, df_teste: pd.DataFrame) -> np.ndarray:
    """Regressão linear rendimento ~ ano, por município.

    Cai de volta para a média histórica se o município tiver menos de 3
    anos de treino (poucos pontos para ajustar uma reta com confiança).
    """
    previsoes = pd.Series(index=df_teste.index, dtype=float)
    for municipio_id, grupo_teste in df_teste.groupby("municipio_id"):
        historico = df_treino[df_treino["municipio_id"] == municipio_id]
        if len(historico) >= 3:
            modelo = LinearRegression().fit(historico[["ano"]], historico[TARGET_COL])
            pred = modelo.predict(grupo_teste[["ano"]])
        else:
            pred = grupo_teste["rendimento_medio_historico_kg_ha"].to_numpy()
        previsoes.loc[grupo_teste.index] = pred
    return previsoes.to_numpy()


def ajustar_ridge(df_treino: pd.DataFrame, feature_cols: list[str] = FEATURE_COLS):
    pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    pipe.fit(_preparar_x(df_treino, df_treino, feature_cols), df_treino[TARGET_COL])
    return pipe


def prever_ridge(
    df_treino: pd.DataFrame, df_teste: pd.DataFrame, feature_cols: list[str] = FEATURE_COLS
):
    modelo = ajustar_ridge(df_treino, feature_cols)
    return modelo.predict(_preparar_x(df_teste, df_treino, feature_cols))


def ajustar_lightgbm(df_treino: pd.DataFrame, feature_cols: list[str] = FEATURE_COLS, **kwargs):
    import lightgbm as lgb

    parametros = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 5,
        "min_child_samples": 10,
        "random_state": 0,
        "verbosity": -1,
        **kwargs,
    }
    modelo = lgb.LGBMRegressor(**parametros)
    modelo.fit(df_treino[feature_cols], df_treino[TARGET_COL])
    return modelo


def prever_lightgbm(
    df_treino: pd.DataFrame, df_teste: pd.DataFrame, feature_cols: list[str] = FEATURE_COLS
):
    modelo = ajustar_lightgbm(df_treino, feature_cols)
    return modelo.predict(df_teste[feature_cols])


MODELOS = {
    "media_historica": baseline_media_historica,
    "media_mais_tendencia": baseline_media_mais_tendencia,
    "ridge": prever_ridge,
    "lightgbm": prever_lightgbm,
}


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def validacao_temporal_expansiva(
    df: pd.DataFrame,
    modelos: dict = MODELOS,
    ano_minimo_treino: int = ANO_MINIMO_TREINO,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Para cada ano da série, treina só com anos anteriores e testa nele.

    Retorna (metricas, previsoes): métricas agregadas por modelo/ano, e as
    previsões linha a linha (para análise dos piores erros depois).
    """
    anos = sorted(df["ano"].unique())
    linhas_metricas = []
    previsoes_todas = []

    for ano_teste in anos:
        df_treino = df[df["ano"] < ano_teste]
        df_teste = df[df["ano"] == ano_teste]
        if df_treino["ano"].nunique() < ano_minimo_treino or df_teste.empty:
            continue

        for nome_modelo, funcao in modelos.items():
            pred = funcao(df_treino, df_teste)
            y_teste = df_teste[TARGET_COL].to_numpy()
            linhas_metricas.append(
                {
                    "modelo": nome_modelo,
                    "ano": ano_teste,
                    "rmse": rmse(y_teste, pred),
                    "r2_dentro_do_ano": r2_score(y_teste, pred) if len(y_teste) > 1 else np.nan,
                    "n": len(y_teste),
                }
            )
            previsoes_todas.append(
                pd.DataFrame(
                    {
                        "municipio_id": df_teste["municipio_id"].to_numpy(),
                        "ano": ano_teste,
                        "modelo": nome_modelo,
                        "y_true": y_teste,
                        "y_pred": pred,
                    }
                )
            )

    metricas = pd.DataFrame(linhas_metricas)
    previsoes = pd.concat(previsoes_todas, ignore_index=True) if previsoes_todas else pd.DataFrame()
    return metricas, previsoes


def resumo_por_modelo(metricas: pd.DataFrame) -> pd.DataFrame:
    """RMSE médio ponderado pelo nº de municípios testados em cada ano."""

    def _agregar(g):
        return pd.Series(
            {
                "rmse_medio": np.average(g["rmse"], weights=g["n"]),
                "r2_medio_dentro_do_ano": g["r2_dentro_do_ano"].mean(),
                "anos_testados": g["ano"].nunique(),
            }
        )

    resumo = metricas.groupby("modelo").apply(_agregar, include_groups=False)
    return resumo.sort_values("rmse_medio")


def piores_erros_por_ano(previsoes: pd.DataFrame, modelo: str = "lightgbm") -> pd.Series:
    sub = previsoes[previsoes["modelo"] == modelo].copy()
    sub["erro_abs"] = (sub["y_true"] - sub["y_pred"]).abs()
    return sub.groupby("ano")["erro_abs"].mean().sort_values(ascending=False)


def importancia_shap(modelo, df: pd.DataFrame, feature_cols: list[str] = FEATURE_COLS) -> pd.Series:
    import shap

    explainer = shap.TreeExplainer(modelo)
    valores = explainer.shap_values(df[feature_cols])
    importancia = pd.Series(np.abs(valores).mean(axis=0), index=feature_cols)
    return importancia.sort_values(ascending=False)


def main() -> None:
    import duckdb

    from soja_rs.data import DUCKDB_PATH

    with duckdb.connect(str(DUCKDB_PATH)) as con:
        features = con.execute("SELECT * FROM features").df()
        pam = con.execute("SELECT * FROM pam_soja_rs").df()

    df = montar_dataset(features, pam)
    intervalo = f"{df['ano'].min()}-{df['ano'].max()}"
    print(f"{len(df)} linhas | {df['municipio_id'].nunique()} municípios | anos {intervalo}")

    metricas, previsoes = validacao_temporal_expansiva(df)
    print(resumo_por_modelo(metricas))

    with duckdb.connect(str(DUCKDB_PATH)) as con:
        con.execute("CREATE OR REPLACE TABLE validacao_metricas AS SELECT * FROM metricas")
        con.execute("CREATE OR REPLACE TABLE validacao_previsoes AS SELECT * FROM previsoes")

    modelo_final = ajustar_lightgbm(df)
    print("\nImportância SHAP (modelo final, treinado em todos os dados):")
    print(importancia_shap(modelo_final, df))

    print("\nPiores anos (erro absoluto médio, LightGBM):")
    print(piores_erros_por_ano(previsoes).head())


if __name__ == "__main__":
    main()
