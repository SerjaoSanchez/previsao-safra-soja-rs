"""Coleta e organização dos dados brutos (Fase 1).

Implementado até agora:
  - Coletor da PAM/IBGE (tabela 5457, API de agregados do IBGE — o mesmo
    motor de dados do SIDRA) parametrizado por UF, produto e anos, com cache
    local em ``data/raw/pam/cache/`` e carga em DuckDB.

Ainda por fazer (ver base.txt, Fase 1):
  - Malha municipal do RS + centroides.
  - Coletor NASA POWER com cache local e ``time.sleep`` entre chamadas.
  - Série mensal do índice ONI (NOAA).

Rode com ``make data`` ou ``python -m soja_rs.data``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import duckdb
import pandas as pd
import requests

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
    df = collect_pam()
    load_duckdb(df, "pam_soja_rs")
    _print_sanity_summary(df)


if __name__ == "__main__":
    main()
