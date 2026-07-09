"""
tests/conftest.py
Añade la raíz del repo a sys.path para que `from core.metrics import ...`
y `from core.parsing import ...` funcionen al correr `pytest` desde cualquier cwd.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
