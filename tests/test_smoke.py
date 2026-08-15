"""Teste placeholder — confirma que o pacote importa e o CI está funcionando.

Conforme cada feature ganhar uma função em src/soja_rs/features.py, adicione
um teste correspondente aqui (ex.: série de 30 dias sem chuva deve devolver
sequência seca = 30).
"""

import soja_rs


def test_package_imports():
    assert soja_rs.__version__ == "0.1.0"
