import os
import sys
import importlib.util

SCRIPT_DIR = os.path.join(os.path.dirname(__file__), '..', 'library', 'scripts_utiles')
SCRIPT_PATH = os.path.join(SCRIPT_DIR, 'analisis_descriptivo.py')

class AnalisisResult:
    def __init__(self):
        self.metricas = {}
        self.figuras = []
        self.df = None
        self.stats = {}
        self.audit = {}
        self.config = {}
        self.output_pdf = None

    def _extract_from_namespace(self, ns):
        """Extract computed metrics from the executed module's namespace."""
        self.config = ns.get('CONFIG', self.config)
        self.df = ns.get('df', None)
        self.stats = ns.get('stats_temporales', {})
        self.audit = ns.get('audit', {})
        self.metricas['cagr'] = ns.get('ret_anual', None)
        self.metricas['vol_anual'] = ns.get('vol_anual', None)
        self.metricas['sharpe'] = ns.get('sharpe', None)
        self.metricas['max_dd'] = ns.get('max_dd', None)
        self.metricas['hur medio'] = ns.get('hurst_mean', None)
        self.metricas['er medio'] = ns.get('er_mean', None)
        return self


def analizar(config, output_pdf=None):
    """
    Run analysis with the given config.

    Parameters
    ----------
    config : dict
        Must contain 'nombre', 'tf', 'activo', 'input_path'
    output_pdf : str or None
        Path to save PDF, or None to skip PDF generation.

    Returns
    -------
    AnalisisResult or None
        If output_pdf is given, returns None (runs as subprocess).
        Otherwise returns AnalisisResult with computed metrics.
    """
    if output_pdf:
        return _run_subprocess(config, output_pdf)

    cfg = dict(config)
    cfg.setdefault('interactive', False)

    import runpy
    init_globs = {
        '_CONFIG_READY': True,
        'CONFIG': cfg,
    }

    old_path = list(sys.path)
    sys.path.insert(0, SCRIPT_DIR)
    try:
        globs = runpy.run_path(SCRIPT_PATH, init_globals=init_globs, run_name='__main__')
    finally:
        sys.path = old_path

    result = AnalisisResult()
    result._extract_from_namespace(globs)
    return result


def _run_subprocess(config, output_pdf):
    """Run the original script as a subprocess (legacy path)."""
    import subprocess
    env = os.environ.copy()
    env['CONFIG_NOMBRE'] = config.get('nombre', '')
    env['CONFIG_TF'] = config.get('tf', '')
    env['CONFIG_ACTIVO'] = config.get('activo', '')
    env['CONFIG_INPUT'] = config.get('input_path', '')
    env['OUTPUT_PDF'] = output_pdf
    env['MPLBACKEND'] = 'Agg'
    subprocess.run([sys.executable, SCRIPT_PATH], env=env)
    return None
