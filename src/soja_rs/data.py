"""Coleta e organização dos dados brutos (Fase 1).

Coletores implementados:
  - PAM/IBGE (tabela 5457) — rendimento, área e produção de soja por
    município do RS.
  - Malha municipal do RS + centroides (para consultar a NASA POWER).
  - NASA POWER — clima diário em grade por centroide municipal.
  - Índice ONI (NOAA) — El Niño/La Niña, defasado.

Tudo com cache local em ``data/raw/`` e carga final em DuckDB.

Rode com ``make data`` ou ``python -m soja_rs.data``.
"""

from __future__ import annotations

import io
import json
import socket
import time
from pathlib import Path

import duckdb
import pandas as pd
import requests
import urllib3.util.connection as _urllib3_connection

# Alguns hosts (ex. NASA POWER, atrás de CloudFront) anunciam endereço IPv6
# que não é roteável neste ambiente; requests/urllib3 tenta IPv6 primeiro e
# trava até o timeout em vez de cair para IPv4 rapidamente (o `curl` da CLI
# não sofre disso porque prioriza IPv4). Força IPv4 para todo o módulo.
_urllib3_connection.allowed_gai_family = lambda: socket.AF_INET

RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")

DUCKDB_PATH = PROCESSED_DIR / "soja_rs.duckdb"

# API de agregados do IBGE (mesmo backend de dados do SIDRA clássico,
# apisidra.ibge.gov.br, mas com JSON mais limpo e IDs de município nativos).
IBGE_AGREGADOS_URL = "https://servicodados.ibge.gov.br/api/v3/agregados"

TABELA_PAM = 5457  # PAM: área, produção, rendimento e valor por município
CLASSIFICACAO_PRODUTO = 782  # "Produto das lavouras temporárias e permanentes"
UF_RS = 43
PRODUTO_SOJA = 40124  # "Soja (em grão)"

# variável IBGE -> nome de coluna tidy
VARIAVEIS_PAM = {
    8331: "area_plantada_ha",
    216: "area_colhida_ha",
    214: "quantidade_produzida_t",
    112: "rendimento_medio_kg_ha",
    215: "valor_producao_mil_reais",
}

# Códigos de valor ausente/especial usados pelo IBGE nas séries do SIDRA.
_MISSING_CODES = {"-": 0.0, "..": None, "...": None, "X": None}


def _to_float(raw: str) -> float | None:
    if raw in _MISSING_CODES:
        return _MISSING_CODES[raw]
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _year_chunks(ano_inicio: int, ano_fim: int, tamanho: int = 10):
    """A API rejeita (HTTP 500) uma janela grande demais de anos de uma vez."""
    inicio = ano_inicio
    while inicio <= ano_fim:
        fim = min(inicio + tamanho - 1, ano_fim)
        yield inicio, fim
        inicio = fim + 1


def _fetch_pam_chunk(
    ano_inicio: int,
    ano_fim: int,
    uf: int,
    produto: int,
    variaveis: dict[int, str],
    cache_dir: Path,
    sleep_seconds: float = 1.0,
) -> dict:
    """Busca um intervalo de anos da tabela 5457, com cache local em disco."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    variaveis_str = ",".join(str(v) for v in variaveis)
    cache_file = cache_dir / f"pam_{TABELA_PAM}_uf{uf}_produto{produto}_{ano_inicio}-{ano_fim}.json"

    if cache_file.exists():
        return json.loads(cache_file.read_text())

    url = (
        f"{IBGE_AGREGADOS_URL}/{TABELA_PAM}/periodos/{ano_inicio}-{ano_fim}"
        f"/variaveis/{variaveis_str}"
        f"?localidades=N6[N3[{uf}]]&classificacao={CLASSIFICACAO_PRODUTO}[{produto}]"
    )
    response = requests.get(url, headers={"User-Agent": "soja-rs-portfolio/0.1"}, timeout=60)
    response.raise_for_status()
    payload = response.json()

    cache_file.write_text(json.dumps(payload, ensure_ascii=False))
    time.sleep(sleep_seconds)
    return payload


def _parse_pam_response(payload: list[dict], variaveis: dict[int, str]) -> pd.DataFrame:
    """Achata a resposta aninhada da API num DataFrame tidy (long format)."""
    rows = []
    for variavel_bloco in payload:
        coluna = variaveis[int(variavel_bloco["id"])]
        for resultado in variavel_bloco["resultados"]:
            for serie in resultado["series"]:
                municipio_id = serie["localidade"]["id"]
                municipio_nome = serie["localidade"]["nome"].removesuffix(" - RS")
                for ano, valor in serie["serie"].items():
                    rows.append(
                        {
                            "municipio_id": municipio_id,
                            "municipio_nome": municipio_nome,
                            "ano": int(ano),
                            "variavel": coluna,
                            "valor": _to_float(valor),
                        }
                    )
    long_df = pd.DataFrame(rows)
    # pivot (não pivot_table): cada combinação município/ano/variável é única
    # aqui, e pivot_table descartaria linhas cujo único valor fosse NaN.
    return long_df.pivot(
        index=["municipio_id", "municipio_nome", "ano"],
        columns="variavel",
        values="valor",
    ).reset_index()


def collect_pam(
    uf: int = UF_RS,
    produto: int = PRODUTO_SOJA,
    ano_inicio: int = 1974,
    ano_fim: int = 2024,
    variaveis: dict[int, str] = VARIAVEIS_PAM,
    raw_dir: Path = RAW_DIR,
) -> pd.DataFrame:
    """Coleta a série completa de PAM/IBGE (tabela 5457) para um produto e UF.

    Município é identificado pelo código IBGE de 7 dígitos
    (``municipio_id``) — nunca use o nome como chave, municípios mudam de
    nome e há homônimos entre estados.
    """
    cache_dir = raw_dir / "pam" / "cache"
    frames = []
    for inicio, fim in _year_chunks(ano_inicio, ano_fim):
        payload = _fetch_pam_chunk(inicio, fim, uf, produto, variaveis, cache_dir)
        frames.append(_parse_pam_response(payload, variaveis))

    df = pd.concat(frames, ignore_index=True).sort_values(["municipio_id", "ano"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_dir / "pam_soja_rs.csv", index=False)
    return df


IBGE_MALHAS_URL = "https://servicodados.ibge.gov.br/api/v3/malhas/estados"


def collect_malha_centroides(uf: int = UF_RS, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Baixa a malha municipal do RS (um único request) e calcula centroides.

    Os centroides servem de ponto de consulta para a NASA POWER — cada
    município vira um par (lon, lat).
    """
    import geopandas as gpd

    malha_dir = raw_dir / "malha"
    malha_dir.mkdir(parents=True, exist_ok=True)
    geojson_path = malha_dir / f"rs_{uf}_municipios.geojson"

    if not geojson_path.exists():
        url = f"{IBGE_MALHAS_URL}/{uf}?intrarregiao=municipio&formato=application/vnd.geo+json"
        response = requests.get(url, headers={"User-Agent": "soja-rs-portfolio/0.1"}, timeout=90)
        response.raise_for_status()
        geojson_path.write_bytes(response.content)

    gdf = gpd.read_file(geojson_path)
    # Centroide em CRS geográfico (graus) distorce; projeta para SIRGAS 2000 /
    # Brazil Polyconic (metros), calcula o centroide e volta para lon/lat.
    centroides = gdf.geometry.to_crs("EPSG:5880").centroid.to_crs("EPSG:4674")
    df = pd.DataFrame(
        {
            "municipio_id": gdf["codarea"].astype(str),
            "lon": centroides.x,
            "lat": centroides.y,
        }
    ).sort_values("municipio_id")

    raw_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_dir / "municipios_rs_centroides.csv", index=False)
    return df


NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
NASA_POWER_START_YEAR = 1981  # cobertura da grade da NASA POWER começa aqui
NASA_POWER_PARAMETROS = {
    "T2M_MAX": "tmax_c",
    "T2M_MIN": "tmin_c",
    "PRECTOTCORR": "precip_mm",
    "ALLSKY_SFC_SW_DWN": "radiacao_mj_m2",
    "RH2M": "umidade_relativa_pct",
}


def _fetch_nasa_power_point(
    municipio_id: str,
    lon: float,
    lat: float,
    ano_inicio: int,
    ano_fim: int,
    cache_dir: Path,
    sleep_seconds: float = 1.0,
) -> dict:
    """Busca a série diária de um ponto na NASA POWER, com cache local."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"power_{municipio_id}_{ano_inicio}-{ano_fim}.json"

    if cache_file.exists():
        return json.loads(cache_file.read_text())

    params = {
        "parameters": ",".join(NASA_POWER_PARAMETROS),
        "community": "AG",
        "longitude": f"{lon:.4f}",
        "latitude": f"{lat:.4f}",
        "start": f"{ano_inicio}0101",
        "end": f"{ano_fim}1231",
        "format": "JSON",
    }
    response = requests.get(
        NASA_POWER_URL, params=params, headers={"User-Agent": "soja-rs-portfolio/0.1"}, timeout=120
    )
    response.raise_for_status()
    payload = response.json()

    cache_file.write_text(json.dumps(payload, ensure_ascii=False))
    time.sleep(sleep_seconds)
    return payload


def _parse_nasa_power_response(payload: dict, municipio_id: str) -> pd.DataFrame:
    parametros = payload["properties"]["parameter"]
    df = pd.DataFrame({coluna: parametros[var] for var, coluna in NASA_POWER_PARAMETROS.items()})
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "data"
    df = df.reset_index()
    df.insert(0, "municipio_id", municipio_id)
    # NASA POWER usa -999 como sentinela de dado ausente.
    valor_cols = list(NASA_POWER_PARAMETROS.values())
    df[valor_cols] = df[valor_cols].replace(-999, pd.NA)
    return df


def collect_clima(
    centroides: pd.DataFrame,
    ano_inicio: int = NASA_POWER_START_YEAR,
    ano_fim: int = 2024,
    raw_dir: Path = RAW_DIR,
    sleep_seconds: float = 0.5,
) -> pd.DataFrame:
    """Coleta o clima diário da NASA POWER para cada município (centroide).

    Um município = uma chamada com a série inteira (o range completo
    1981-2024 é rápido num único request), cacheada em disco. O
    ``sleep_seconds`` entre chamadas evita sobrecarregar a API pública.
    """
    cache_dir = raw_dir / "power" / "cache"
    frames = []
    total = len(centroides)
    for i, row in enumerate(centroides.itertuples(), start=1):
        payload = _fetch_nasa_power_point(
            row.municipio_id, row.lon, row.lat, ano_inicio, ano_fim, cache_dir, sleep_seconds
        )
        frames.append(_parse_nasa_power_response(payload, row.municipio_id))
        if i % 25 == 0 or i == total:
            print(f"  NASA POWER: {i}/{total} municípios")

    df = pd.concat(frames, ignore_index=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(raw_dir / "clima_diario_rs.parquet", index=False)
    return df


NOAA_ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"


def collect_oni(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Baixa o índice ONI (El Niño/La Niña) trimestral da NOAA.

    A coluna ``son_anomalia`` (Set-Out-Nov) é o valor usado como feature: é
    o ONI mais recente disponível antes do plantio da soja no RS.
    """
    cache_file = raw_dir / "oni.txt"
    if cache_file.exists():
        texto = cache_file.read_text()
    else:
        headers = {"User-Agent": "soja-rs-portfolio/0.1"}
        response = requests.get(NOAA_ONI_URL, headers=headers, timeout=30)
        response.raise_for_status()
        texto = response.text
        raw_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(texto)

    df = pd.read_csv(io.StringIO(texto), sep=r"\s+")
    df.columns = ["temporada", "ano", "total", "anomalia"]
    son = df.query("temporada == 'SON'")[["ano", "anomalia"]]
    son = son.rename(columns={"anomalia": "son_anomalia"})
    son.to_csv(raw_dir / "oni_son.csv", index=False)
    return son


def load_duckdb(df: pd.DataFrame, table_name: str, db_path: Path = DUCKDB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(db_path)) as con:
        con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df")


def _print_sanity_summary(df: pd.DataFrame) -> None:
    n_municipios = df["municipio_id"].nunique()
    n_anos = df["ano"].nunique()
    intervalo = f"{df['ano'].min()}-{df['ano'].max()}"
    print(f"{len(df)} linhas | {n_municipios} municípios | {n_anos} anos ({intervalo})")
    for col in VARIAVEIS_PAM.values():
        pct_na = df[col].isna().mean() * 100
        print(f"  {col}: {pct_na:.1f}% ausente")


def main() -> None:
    print("PAM/IBGE (rendimento de soja)...")
    pam = collect_pam()
    load_duckdb(pam, "pam_soja_rs")
    _print_sanity_summary(pam)

    print("\nMalha municipal + centroides...")
    centroides = collect_malha_centroides()
    load_duckdb(centroides, "municipios_centroides")
    print(f"{len(centroides)} municípios")

    print("\nClima diário (NASA POWER)...")
    clima = collect_clima(centroides)
    load_duckdb(clima, "clima_diario")
    print(f"{len(clima)} linhas ({clima['data'].min().date()} a {clima['data'].max().date()})")

    print("\nÍndice ONI (NOAA)...")
    oni = collect_oni()
    load_duckdb(oni, "oni_son")
    print(f"{len(oni)} anos")


if __name__ == "__main__":
    main()
