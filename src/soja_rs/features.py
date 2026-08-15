"""Engenharia de features fenológicas (Fase 2).

Uma linha por (município, ano-safra). ``ano`` segue a convenção do PAM/IBGE:
é o ano de colheita. Como o corte de dados é 31 de janeiro, a janela
climática disponível vai de novembro do ano anterior (semeadura) a janeiro
do próprio ano (floração + início do enchimento de grão) — fevereiro em
diante (fim do enchimento) não é observável e por isso não entra em nenhuma
feature; ver README, seção "Limitações".

Cada função pura abaixo (graus_dia, maior_sequencia_seca, dias_tmax_acima,
et0_hargreaves) é testável isoladamente com séries sintéticas, sem precisar
dos dados reais coletados na Fase 1.

Rode com ``make features`` ou ``python -m soja_rs.features``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BASE_GDD = 10.0
LIMIAR_SECA_MM = 1.0
LIMIAR_TMAX_QUENTE = 32.0
NORMAL_ANOS = 30

_JANELA_POR_MES = {11: "semeadura", 12: "vegetativo", 1: "reprodutivo"}
_DESLOCAMENTO_POR_MES = {11: 1, 12: 1, 1: 0}  # nov/dez contam para o ano-safra seguinte


def graus_dia(tmax: pd.Series, tmin: pd.Series, base: float = BASE_GDD) -> pd.Series:
    """Graus-dia acumulados por dia (base fixa, sem truncamento de tmin/tmax)."""
    media = (tmax + tmin) / 2
    return (media - base).clip(lower=0)


def maior_sequencia_seca(precip: pd.Series, limiar: float = LIMIAR_SECA_MM) -> int:
    """Maior número de dias consecutivos com chuva abaixo do limiar."""
    seco = (precip < limiar).astype(int)
    if seco.sum() == 0:
        return 0
    grupos = (seco != seco.shift()).cumsum()
    return int(seco.groupby(grupos).sum().max())


def dias_tmax_acima(tmax: pd.Series, limiar: float = LIMIAR_TMAX_QUENTE) -> int:
    """Número de dias com temperatura máxima acima do limiar."""
    return int((tmax > limiar).sum())


def et0_hargreaves(tmax: pd.Series, tmin: pd.Series, datas: pd.Series, lat) -> pd.Series:
    """ET0 diária (mm/dia), método de Hargreaves-Samani.

    Depende só de temperatura e da radiação extraterrestre astronômica
    (função de latitude e dia do ano) — não da radiação medida pela NASA
    POWER, que tem falhas na série (~7% ausente).
    """
    doy = pd.DatetimeIndex(datas).dayofyear.to_numpy()
    lat_rad = np.radians(np.asarray(lat, dtype=float))
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    delta = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(lat_rad) * np.tan(delta), -1, 1))
    ra_mj = (
        (24 * 60 / np.pi)
        * 0.0820
        * dr
        * (ws * np.sin(lat_rad) * np.sin(delta) + np.cos(lat_rad) * np.cos(delta) * np.sin(ws))
    )
    ra_mm = 0.408 * ra_mj  # MJ/m²/dia -> mm/dia equivalentes de evaporação
    tmean = (tmax.to_numpy() + tmin.to_numpy()) / 2
    amplitude = np.clip(tmax.to_numpy() - tmin.to_numpy(), 0, None)
    et0 = 0.0023 * ra_mm * np.sqrt(amplitude) * (tmean + 17.8)
    return pd.Series(np.clip(et0, 0, None), index=tmax.index)


def build_features(
    clima_diario: pd.DataFrame,
    centroides: pd.DataFrame,
    oni_son: pd.DataFrame,
    pam: pd.DataFrame,
    normal_anos: int = NORMAL_ANOS,
) -> pd.DataFrame:
    """Monta a tabela de features fenológicas, uma linha por município-ano-safra."""
    coords = centroides[["municipio_id", "lat", "lon"]]
    clima = clima_diario.merge(coords, on="municipio_id", how="inner")
    mes = clima["data"].dt.month
    clima = clima[mes.isin([11, 12, 1])].copy()
    mes = clima["data"].dt.month

    clima["janela"] = mes.map(_JANELA_POR_MES)
    clima["ano_safra"] = clima["data"].dt.year + mes.map(_DESLOCAMENTO_POR_MES)
    clima["gdd_dia"] = graus_dia(clima["tmax_c"], clima["tmin_c"])
    clima["et0_dia"] = et0_hargreaves(clima["tmax_c"], clima["tmin_c"], clima["data"], clima["lat"])
    clima["dia_quente"] = clima["tmax_c"] > LIMIAR_TMAX_QUENTE

    agregados = (
        clima.groupby(["municipio_id", "ano_safra", "janela"])
        .agg(
            gdd=("gdd_dia", "sum"),
            precip_mm=("precip_mm", "sum"),
            radiacao_mj_m2=("radiacao_mj_m2", "sum"),
            et0_mm=("et0_dia", "sum"),
            dias_quentes=("dia_quente", "sum"),
        )
        .reset_index()
    )

    largo = agregados.pivot(index=["municipio_id", "ano_safra"], columns="janela")
    largo.columns = [f"{metrica}_{janela}" for metrica, janela in largo.columns]
    largo = largo.reset_index()

    reprodutivo = clima[clima["janela"] == "reprodutivo"]
    seca = (
        reprodutivo.groupby(["municipio_id", "ano_safra"])["precip_mm"]
        .apply(maior_sequencia_seca)
        .rename("maior_seca_reprodutivo_dias")
        .reset_index()
    )
    largo = largo.merge(seca, on=["municipio_id", "ano_safra"], how="left")

    largo["balanco_hidrico_reprodutivo_mm"] = (
        largo["precip_mm_reprodutivo"] - largo["et0_mm_reprodutivo"]
    )

    # Janelas incompletas nas bordas da série (1981 sem nov/dez de 1980;
    # 2025 sem janeiro) ficam com NaN nas colunas essenciais — descarta.
    essenciais = ["gdd_semeadura", "gdd_vegetativo", "gdd_reprodutivo"]
    largo = largo.dropna(subset=essenciais)
    largo = largo.sort_values(["municipio_id", "ano_safra"]).reset_index(drop=True)

    # Anomalia de precipitação: normal trailing (até `normal_anos` anos
    # anteriores) por município — nunca olha para o próprio ano ou o futuro.
    normal = largo.groupby("municipio_id")["precip_mm_reprodutivo"].transform(
        lambda s: s.rolling(normal_anos, min_periods=5).mean().shift(1)
    )
    largo["anomalia_precip_reprodutivo_mm"] = largo["precip_mm_reprodutivo"] - normal

    # ONI defasado: SON do ano anterior ao ano-safra (disponível antes do plantio).
    oni = oni_son.rename(columns={"ano": "ano_oni", "son_anomalia": "oni_lag_son"})
    largo = largo.merge(oni, left_on=largo["ano_safra"] - 1, right_on="ano_oni", how="left")
    largo = largo.drop(columns="ano_oni")

    # Rendimento histórico do município: média expandida, só anos
    # anteriores (evita vazamento de informação futura).
    pam_ordenado = pam.sort_values(["municipio_id", "ano"]).copy()
    pam_ordenado["rendimento_medio_historico_kg_ha"] = pam_ordenado.groupby("municipio_id")[
        "rendimento_medio_kg_ha"
    ].transform(lambda s: s.shift().expanding().mean())
    largo = largo.merge(
        pam_ordenado[["municipio_id", "ano", "rendimento_medio_historico_kg_ha"]],
        left_on=["municipio_id", "ano_safra"],
        right_on=["municipio_id", "ano"],
        how="left",
    ).drop(columns="ano")

    largo = largo.merge(centroides[["municipio_id", "lat", "lon"]], on="municipio_id", how="left")

    colunas = [
        "municipio_id",
        "ano_safra",
        "gdd_semeadura",
        "gdd_vegetativo",
        "gdd_reprodutivo",
        "precip_mm_semeadura",
        "precip_mm_vegetativo",
        "precip_mm_reprodutivo",
        "anomalia_precip_reprodutivo_mm",
        "maior_seca_reprodutivo_dias",
        "dias_quentes_reprodutivo",
        "radiacao_mj_m2_reprodutivo",
        "balanco_hidrico_reprodutivo_mm",
        "oni_lag_son",
        "rendimento_medio_historico_kg_ha",
        "lat",
        "lon",
    ]
    renomeio = {"ano_safra": "ano", "dias_quentes_reprodutivo": "dias_tmax32_reprodutivo"}
    return largo[colunas].rename(columns=renomeio)


def main() -> None:
    import duckdb

    from soja_rs.data import DUCKDB_PATH

    with duckdb.connect(str(DUCKDB_PATH)) as con:
        clima = con.execute("SELECT * FROM clima_diario").df()
        centroides = con.execute("SELECT * FROM municipios_centroides").df()
        oni = con.execute("SELECT * FROM oni_son").df()
        pam = con.execute("SELECT * FROM pam_soja_rs").df()

    features = build_features(clima, centroides, oni, pam)

    with duckdb.connect(str(DUCKDB_PATH)) as con:
        con.execute("CREATE OR REPLACE TABLE features AS SELECT * FROM features")

    n_municipios = features["municipio_id"].nunique()
    intervalo = f"{features['ano'].min()}-{features['ano'].max()}"
    print(f"{len(features)} linhas | {n_municipios} municípios | anos {intervalo}")
    print(features.isna().mean().round(3))


if __name__ == "__main__":
    main()
