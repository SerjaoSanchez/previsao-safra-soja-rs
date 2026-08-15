# Previsão de safra de soja no RS

**Pergunta do projeto:** dá para prever o rendimento médio (kg/ha) de soja por
município do Rio Grande do Sul, para o ano-safra seguinte, usando apenas
informação climática disponível até 31 de janeiro?

## O que isso significa

Todo ano, produtores, seguradoras e traders de commodities precisam estimar
como vai ser a safra de soja *antes* dela terminar. O Rio Grande do Sul é o
estado brasileiro com maior variação de produtividade de um ano para o outro,
porque a soja gaúcha é de sequeiro (não irrigada) e depende diretamente do
regime de chuvas — especialmente de eventos La Niña, que historicamente
coincidem com quebras de safra (2004/05, 2011/12, 2021/22, 2022/23).

Este projeto usa dados públicos de clima, produção agrícola e o índice
climático ONI (El Niño/La Niña) para treinar um modelo que prevê o rendimento
municipal *antes* da fase mais crítica do ciclo da soja (enchimento de grão)
terminar — usando só o que já se sabe até 31 de janeiro, sem espiar o futuro.

> Resultado, gráficos e comparação com o baseline entram aqui assim que o
> modelo estiver treinado (Fase 3).

## Escopo

- **Cultura:** soja (sequeiro — o sinal de estresse hídrico é limpo, diferente
  do arroz irrigado gaúcho).
- **Recorte geográfico:** municípios do Rio Grande do Sul.
- **Unidade de análise:** município-ano (~450 municípios × ~20 anos).
- **Corte temporal:** apenas dados disponíveis até 31 de janeiro do ano-safra
  — o projeto é previsão de verdade, não explicação retrospectiva.

## Fontes de dados

| Fonte | Uso | Acesso |
|---|---|---|
| PAM/IBGE (tabela 5457) | Alvo: rendimento médio municipal | API SIDRA |
| CONAB | Validação estadual (nunca como alvo) | Séries históricas |
| NASA POWER | Clima diário em grade por centroide municipal | API pública |
| INMET (opcional) | Validação da grade climática contra estações reais | BDMEP |
| ONI (NOAA) | Índice La Niña/El Niño defasado, disponível antes do plantio | CSV público |

## Estrutura do repositório

```
data/
  raw/          # dados brutos baixados (fora do git)
  interim/      # dados em processamento (fora do git)
  processed/    # dados prontos para modelagem (fora do git)
src/soja_rs/    # código do projeto (o notebook é rascunho; isto é o produto)
notebooks/      # exploração e sanidade
tests/          # testes das funções de features/coleta
reports/        # figuras, métricas, saídas de análise
```

## Como rodar

```bash
make data      # baixa e organiza os dados brutos em DuckDB
make features  # gera as features fenológicas
make train     # treina e valida os modelos (baselines + LightGBM)
make app       # roda o app Streamlit localmente
```

## Roadmap

- [x] Fase 0 — Fundação (repositório, ambiente, estrutura)
- [ ] Fase 1 — Dados (PAM/SIDRA, NASA POWER, ONI, DuckDB)
- [ ] Fase 2 — Features fenológicas (GDD, balanço hídrico, ONI defasado)
- [ ] Fase 3 — Modelagem (baselines → Ridge → LightGBM, validação temporal)
- [ ] Fase 4 — Entrega (app Streamlit, CI, README com resultado)

## Limitações

_(preencher ao final — o que foi tentado e não funcionou, vazamentos
encontrados, decisões metodológicas questionáveis)._
