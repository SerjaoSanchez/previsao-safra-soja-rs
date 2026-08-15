"""Modelagem e validação (Fase 3).

Baselines obrigatórios, nesta ordem: média histórica do município → média +
tendência linear temporal → Ridge → LightGBM. Validação sempre temporal
(leave-one-year-out ou expansiva) — nunca K-fold aleatório, que vaza
informação do mesmo ano entre treino e teste.

Rode com ``make train`` ou ``python -m soja_rs.train``.
"""


def main() -> None:
    raise NotImplementedError("Fase 3: baselines e modelo ainda não implementados.")


if __name__ == "__main__":
    main()
