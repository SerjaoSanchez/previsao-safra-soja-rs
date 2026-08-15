"""Coleta e organização dos dados brutos (Fase 1).

Responsabilidades deste módulo, quando implementado:
  - Coletor da PAM/IBGE (tabela 5457, API SIDRA) parametrizado por UF, cultura
    e anos, salvando em ``data/raw/``.
  - Malha municipal do RS + centroides (para consultar a NASA POWER).
  - Coletor NASA POWER com cache local (não rebaixar o que já foi baixado) e
    ``time.sleep`` entre chamadas.
  - Série mensal do índice ONI (NOAA).
  - Carga de tudo em um arquivo DuckDB único.

Rode com ``make data`` ou ``python -m soja_rs.data``.
"""

from pathlib import Path

RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")


def main() -> None:
    raise NotImplementedError("Fase 1: coletores de PAM/SIDRA, NASA POWER e ONI ainda não implementados.")


if __name__ == "__main__":
    main()
