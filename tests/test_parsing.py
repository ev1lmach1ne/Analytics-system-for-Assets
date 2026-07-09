import numpy as np
import pytest

from core.parsing import parse_numero_flexible


@pytest.mark.parametrize("entrada,valor_esperado", [
    ("1234.56", 1234.56),
    ("1.234,56", 1234.56),   # formato europeo (punto=miles, coma=decimal)
    ("1,234.56", 1234.56),   # formato americano (coma=miles, punto=decimal)
    ("1.5K", 1500.0),
    ("2M", 2_000_000.0),
    ("1B", 1_000_000_000.0),
])
def test_parse_numero_flexible_ok(entrada, valor_esperado):
    valor, motivo = parse_numero_flexible(entrada)
    assert motivo == 'ok'
    assert valor == pytest.approx(valor_esperado, abs=1e-9)


@pytest.mark.parametrize("entrada", [np.nan, "", "null", "None", "  "])
def test_parse_numero_flexible_vacio(entrada):
    valor, motivo = parse_numero_flexible(entrada)
    assert motivo == 'vacio'
    assert valor == 0.0


def test_parse_numero_flexible_fallo():
    valor, motivo = parse_numero_flexible("abc123xyz")
    assert motivo == 'fallo'
    assert valor == 0.0
