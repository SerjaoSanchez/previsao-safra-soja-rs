import pandas as pd

from soja_rs.data import _parse_nasa_power_response


def _fake_power_payload():
    return {
        "properties": {
            "parameter": {
                "T2M_MAX": {"20200101": 30.1, "20200102": -999},
                "T2M_MIN": {"20200101": 18.2, "20200102": 17.5},
                "PRECTOTCORR": {"20200101": 0.0, "20200102": 12.4},
                "ALLSKY_SFC_SW_DWN": {"20200101": 25.3, "20200102": 10.1},
                "RH2M": {"20200101": 60.0, "20200102": 88.2},
            }
        }
    }


def test_parse_nasa_power_response_colunas_e_datas():
    df = _parse_nasa_power_response(_fake_power_payload(), "4300034")

    assert list(df.columns) == [
        "municipio_id",
        "data",
        "tmax_c",
        "tmin_c",
        "precip_mm",
        "radiacao_mj_m2",
        "umidade_relativa_pct",
    ]
    assert (df["municipio_id"] == "4300034").all()
    assert df["data"].tolist() == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]


def test_parse_nasa_power_response_sentinela_ausente_vira_na():
    df = _parse_nasa_power_response(_fake_power_payload(), "4300034")
    assert pd.isna(df.loc[df["data"] == "2020-01-02", "tmax_c"].item())
    assert df.loc[df["data"] == "2020-01-01", "tmax_c"].item() == 30.1
