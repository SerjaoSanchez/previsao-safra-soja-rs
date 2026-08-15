from soja_rs.data import VARIAVEIS_PAM, _parse_pam_response, _to_float, _year_chunks


def test_to_float_valor_real():
    assert _to_float("1050") == 1050.0
    assert _to_float("1050.5") == 1050.5


def test_to_float_zero_absoluto():
    # "-" no PAM/IBGE é zero de verdade (não arredondamento), não ausência.
    assert _to_float("-") == 0.0


def test_to_float_ausente():
    assert _to_float("..") is None
    assert _to_float("...") is None
    assert _to_float("X") is None


def test_year_chunks_respeita_tamanho_e_limites():
    chunks = list(_year_chunks(1974, 1999, tamanho=10))
    assert chunks == [(1974, 1983), (1984, 1993), (1994, 1999)]


def test_year_chunks_intervalo_menor_que_tamanho():
    assert list(_year_chunks(2020, 2021, tamanho=10)) == [(2020, 2021)]


def _fake_payload():
    return [
        {
            "id": "112",
            "variavel": "Rendimento médio da produção",
            "resultados": [
                {
                    "series": [
                        {
                            "localidade": {"id": "4300034", "nome": "Aceguá - RS"},
                            "serie": {"2019": "2100", "2020": "1050"},
                        },
                        {
                            "localidade": {"id": "4300570", "nome": "Alto Feliz - RS"},
                            "serie": {"2019": "-", "2020": ".."},
                        },
                    ]
                }
            ],
        }
    ]


def test_parse_pam_response_formato_tidy():
    variaveis = {112: VARIAVEIS_PAM[112]}
    df = _parse_pam_response(_fake_payload(), variaveis)

    assert set(df.columns) == {"municipio_id", "municipio_nome", "ano", "rendimento_medio_kg_ha"}
    assert len(df) == 4

    acegua_2020 = df.query("municipio_id == '4300034' and ano == 2020")
    assert acegua_2020["rendimento_medio_kg_ha"].item() == 1050.0
    assert acegua_2020["municipio_nome"].item() == "Aceguá"

    alto_feliz_2019 = df.query("municipio_id == '4300570' and ano == 2019")
    assert alto_feliz_2019["rendimento_medio_kg_ha"].item() == 0.0

    alto_feliz_2020 = df.query("municipio_id == '4300570' and ano == 2020")
    assert alto_feliz_2020["rendimento_medio_kg_ha"].isna().item()
