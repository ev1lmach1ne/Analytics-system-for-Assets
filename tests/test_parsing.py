import numpy as np
import pandas as pd
import pytest

from core.parsing import parse_numero_flexible, parse_columna_flexible


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


# ── Tests versión vectorizada ──

@pytest.mark.parametrize("entrada,valor_esperado", [
    ("1234.56", 1234.56),
    ("1.234,56", 1234.56),
    ("1,234.56", 1234.56),
    ("1.5K", 1500.0),
    ("2M", 2_000_000.0),
    ("1B", 1_000_000_000.0),
])
def test_parse_columna_flexible_ok(entrada, valor_esperado):
    col = pd.Series([entrada])
    vals, motivos = parse_columna_flexible(col)
    assert motivos.iloc[0] == 'ok'
    assert vals.iloc[0] == pytest.approx(valor_esperado, abs=1e-9)


def test_parse_columna_flexible_vacio():
    col = pd.Series([np.nan, "", "null", "None", "  "])
    vals, motivos = parse_columna_flexible(col)
    assert (motivos == 'vacio').all()
    assert (vals == 0.0).all()


def test_parse_columna_flexible_fallo():
    col = pd.Series(["abc123xyz"])
    vals, motivos = parse_columna_flexible(col)
    assert motivos.iloc[0] == 'fallo'
    assert vals.iloc[0] == 0.0


def test_parse_columna_flexible_mixto():
    col = pd.Series(["1234.56", "1.5K", np.nan, "abc", "2M", ""])
    vals, motivos = parse_columna_flexible(col)
    assert motivos[0] == 'ok'
    assert vals[0] == pytest.approx(1234.56, abs=1e-9)
    assert motivos[1] == 'ok'
    assert vals[1] == pytest.approx(1500.0, abs=1e-9)
    assert motivos[2] == 'vacio'
    assert motivos[3] == 'fallo'
    assert motivos[4] == 'ok'
    assert vals[4] == pytest.approx(2_000_000.0, abs=1e-9)
    assert motivos[5] == 'vacio'


def test_parse_columna_equivalente_a_scalar():
    entradas = ["1234.56", "1.234,56", "1,234.56", "1.5K", "2M", "1B",
                np.nan, "", "null", "abc", "3,14", "0.5k"]
    col = pd.Series(entradas)
    vals_v, mot_v = parse_columna_flexible(col)
    for i, e in enumerate(entradas):
        val_s, mot_s = parse_numero_flexible(e)
        assert mot_v.iloc[i] == mot_s, f"Motivo mismatch en '{e}': {mot_v.iloc[i]} vs {mot_s}"
        assert vals_v.iloc[i] == pytest.approx(val_s, abs=1e-9), f"Valor mismatch en '{e}'"
