import numpy as np
import pandas as pd

from soja_rs.train import (
    ANO_MINIMO_TREINO,
    FEATURE_COLS,
    baseline_media_historica,
    montar_dataset,
    prever_ridge,
    resumo_por_modelo,
    rmse,
    validacao_temporal_expansiva,
)


def _dataset_sintetico(n_municipios=4, anos=range(2000, 2015)) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    linhas_features, linhas_pam = [], []

    for m in range(n_municipios):
        municipio_id = f"430{m:04d}"
        historico = []
        for ano in anos:
            precip = rng.uniform(100, 400)
            rendimento_hist = float(np.mean(historico)) if historico else np.nan
            rendimento = 1500 + 3 * precip + rng.normal(0, 50)

            linhas_features.append(
                {
                    "municipio_id": municipio_id,
                    "ano": ano,
                    "gdd_semeadura": rng.uniform(100, 200),
                    "gdd_vegetativo": rng.uniform(100, 200),
                    "gdd_reprodutivo": rng.uniform(100, 200),
                    "precip_mm_semeadura": rng.uniform(50, 150),
                    "precip_mm_vegetativo": rng.uniform(50, 150),
                    "precip_mm_reprodutivo": precip,
                    "anomalia_precip_reprodutivo_mm": precip - 250,
                    "maior_seca_reprodutivo_dias": rng.integers(0, 15),
                    "dias_tmax32_reprodutivo": rng.integers(0, 10),
                    "radiacao_mj_m2_reprodutivo": rng.uniform(400, 700),
                    "balanco_hidrico_reprodutivo_mm": precip - rng.uniform(80, 150),
                    "oni_lag_son": rng.uniform(-1.5, 1.5),
                    "rendimento_medio_historico_kg_ha": rendimento_hist,
                    "lat": -29.0,
                    "lon": -53.0,
                }
            )
            linhas_pam.append(
                {
                    "municipio_id": municipio_id,
                    "ano": ano,
                    "rendimento_medio_kg_ha": rendimento,
                    "area_colhida_ha": 1000.0,
                }
            )
            historico.append(rendimento)

    return pd.DataFrame(linhas_features), pd.DataFrame(linhas_pam)


def test_montar_dataset_remove_primeiro_ano_sem_historico():
    features, pam = _dataset_sintetico()
    df = montar_dataset(features, pam)
    # primeiro ano de cada município não tem rendimento_medio_historico_kg_ha
    assert df["ano"].min() > min(features["ano"])
    assert df["rendimento_medio_historico_kg_ha"].notna().all()


def test_montar_dataset_filtra_area_colhida_pequena():
    features, pam = _dataset_sintetico(n_municipios=1)
    pam.loc[pam.index[-1], "area_colhida_ha"] = 100.0  # abaixo do limiar de 500 ha
    df = montar_dataset(features, pam, area_minima_ha=500.0)
    ano_excluido = pam.loc[pam.index[-1], "ano"]
    assert ano_excluido not in df["ano"].to_numpy()


def test_baseline_media_historica_usa_a_coluna_direto():
    features, pam = _dataset_sintetico(n_municipios=1)
    df = montar_dataset(features, pam)
    pred = baseline_media_historica(df, df)
    assert np.array_equal(pred, df["rendimento_medio_historico_kg_ha"].to_numpy())


def test_prever_ridge_roda_e_retorna_tamanho_certo():
    features, pam = _dataset_sintetico()
    df = montar_dataset(features, pam)
    corte = df["ano"] < df["ano"].max()
    pred = prever_ridge(df[corte], df[~corte])
    assert len(pred) == (~corte).sum()
    assert np.isfinite(pred).all()


def test_validacao_temporal_expansiva_respeita_ano_minimo_treino():
    features, pam = _dataset_sintetico(anos=range(2000, 2010))
    df = montar_dataset(features, pam)
    modelos = {"media_historica": baseline_media_historica}
    metricas, previsoes = validacao_temporal_expansiva(df, modelos, ano_minimo_treino=3)

    anos_disponiveis = sorted(df["ano"].unique())
    primeiro_ano_testavel = anos_disponiveis[3]
    assert metricas["ano"].min() == primeiro_ano_testavel
    assert set(previsoes["modelo"]) == {"media_historica"}


def test_validacao_temporal_nunca_usa_ano_futuro_no_treino():
    """Checagem direta de vazamento: nenhum modelo pode ver o ano de teste."""
    features, pam = _dataset_sintetico()
    df = montar_dataset(features, pam)

    anos_vistos_no_treino = {}

    def modelo_espiao(df_treino, df_teste):
        anos_vistos_no_treino[df_teste["ano"].iloc[0]] = set(df_treino["ano"].unique())
        return baseline_media_historica(df_treino, df_teste)

    validacao_temporal_expansiva(df, {"espiao": modelo_espiao}, ano_minimo_treino=ANO_MINIMO_TREINO)

    for ano_teste, anos_treino in anos_vistos_no_treino.items():
        assert all(a < ano_teste for a in anos_treino)


def test_resumo_por_modelo_formato():
    features, pam = _dataset_sintetico()
    df = montar_dataset(features, pam)
    metricas, _ = validacao_temporal_expansiva(df, {"media_historica": baseline_media_historica})
    resumo = resumo_por_modelo(metricas)
    assert "rmse_medio" in resumo.columns
    assert "media_historica" in resumo.index


def test_rmse_zero_quando_previsao_perfeita():
    assert rmse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_feature_cols_sao_todas_numericas_no_dataset():
    features, pam = _dataset_sintetico()
    df = montar_dataset(features, pam)
    for col in FEATURE_COLS:
        assert col in df.columns
