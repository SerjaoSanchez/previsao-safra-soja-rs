import numpy as np
import pandas as pd

from soja_rs.features import (
    build_features,
    dias_tmax_acima,
    et0_hargreaves,
    graus_dia,
    maior_sequencia_seca,
)


def test_graus_dia_abaixo_da_base_e_zero():
    tmax = pd.Series([12.0, 8.0])
    tmin = pd.Series([8.0, 4.0])
    # médias: 10 (== base, gdd=0) e 6 (< base, gdd=0 por causa do clip)
    resultado = graus_dia(tmax, tmin, base=10.0)
    assert resultado.tolist() == [0.0, 0.0]


def test_graus_dia_acima_da_base():
    tmax = pd.Series([30.0])
    tmin = pd.Series([20.0])
    # média 25, base 10 -> gdd = 15
    assert graus_dia(tmax, tmin, base=10.0).iloc[0] == 15.0


def test_maior_sequencia_seca_30_dias_sem_chuva():
    precip = pd.Series([0.0] * 30)
    assert maior_sequencia_seca(precip) == 30


def test_maior_sequencia_seca_com_chuva_no_meio():
    precip = pd.Series([0.0] * 10 + [5.0] + [0.0] * 15)
    assert maior_sequencia_seca(precip) == 15


def test_maior_sequencia_seca_sem_dias_secos():
    precip = pd.Series([10.0, 20.0, 5.0])
    assert maior_sequencia_seca(precip) == 0


def test_dias_tmax_acima_conta_corretamente():
    tmax = pd.Series([30.0, 33.0, 35.0, 31.0])
    assert dias_tmax_acima(tmax, limiar=32.0) == 2


def test_et0_hargreaves_produz_valores_positivos_e_plausiveis():
    datas = pd.Series(pd.date_range("2020-01-01", periods=31))
    tmax = pd.Series([30.0] * 31)
    tmin = pd.Series([18.0] * 31)
    et0 = et0_hargreaves(tmax, tmin, datas, lat=-29.0)
    assert (et0 > 0).all()
    # ET0 diária tipicamente entre 1 e 10 mm/dia em climas subtropicais
    assert et0.between(1, 10).all()


def _clima_sintetico(municipio_id: str, anos: list[int]) -> pd.DataFrame:
    """Série diária cobrindo nov(ano-1)-dez(ano-1)-jan(ano) para cada ano."""
    partes = []
    for ano in anos:
        datas = pd.date_range(f"{ano - 1}-11-01", f"{ano}-01-31", freq="D")
        rng = np.random.default_rng(ano)
        partes.append(
            pd.DataFrame(
                {
                    "municipio_id": municipio_id,
                    "data": datas,
                    "tmax_c": rng.uniform(25, 34, len(datas)),
                    "tmin_c": rng.uniform(14, 20, len(datas)),
                    "precip_mm": rng.uniform(0, 15, len(datas)),
                    "radiacao_mj_m2": rng.uniform(15, 28, len(datas)),
                    "umidade_relativa_pct": rng.uniform(50, 90, len(datas)),
                }
            )
        )
    return pd.concat(partes, ignore_index=True)


def test_build_features_formato_e_sem_vazamento():
    clima = _clima_sintetico("4300034", anos=[2019, 2020, 2021])
    centroides = pd.DataFrame({"municipio_id": ["4300034"], "lat": [-29.0], "lon": [-53.0]})
    oni = pd.DataFrame({"ano": [2018, 2019, 2020], "son_anomalia": [0.5, -0.7, -0.9]})
    pam = pd.DataFrame(
        {
            "municipio_id": ["4300034"] * 3,
            "ano": [2019, 2020, 2021],
            "rendimento_medio_kg_ha": [2000.0, 2500.0, 1800.0],
        }
    )

    features = build_features(clima, centroides, oni, pam)

    assert set(features["ano"]) == {2019, 2020, 2021}
    assert (features["gdd_reprodutivo"] > 0).all()

    # ONI defasado: ano-safra 2020 usa o SON de 2019 (ano-1), não o de 2020.
    linha_2020 = features.query("ano == 2020").iloc[0]
    assert linha_2020["oni_lag_son"] == -0.7

    # rendimento histórico de 2021 é a média só de 2019 e 2020 (sem 2021).
    linha_2021 = features.query("ano == 2021").iloc[0]
    assert linha_2021["rendimento_medio_historico_kg_ha"] == 2250.0

    # 2019 é o primeiro ano da série: não há histórico anterior.
    linha_2019 = features.query("ano == 2019").iloc[0]
    assert pd.isna(linha_2019["rendimento_medio_historico_kg_ha"])
