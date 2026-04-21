# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller spec for Random DFS Graph Analyzer
#
# Build on Windows:
#   pyinstaller random_dfs_analyzer.spec
#
# Output: dist/RandomDFSAnalyzer/RandomDFSAnalyzer.exe  (one-dir)
#

from PyInstaller.utils.hooks import collect_submodules, collect_data_files
import sys, os

block_cipher = None

# ── collect hidden imports ────────────────────────────────────────────────────
hidden = []
hidden += collect_submodules("dfs_analyzer")
hidden += collect_submodules("scipy")
hidden += collect_submodules("scipy.optimize")
hidden += collect_submodules("scipy.stats")
hidden += collect_submodules("matplotlib")
hidden += collect_submodules("matplotlib.backends")
hidden += ["matplotlib.backends.backend_qtagg"]
hidden += collect_submodules("pandas")
hidden += collect_submodules("networkx")
hidden += collect_submodules("PyQt6")
hidden += ["PyQt6.QtCore", "PyQt6.QtGui", "PyQt6.QtWidgets"]

# ── data files ────────────────────────────────────────────────────────────────
datas = []
datas += collect_data_files("matplotlib")
datas += collect_data_files("scipy")
# include the whole dfs_analyzer package as source
datas += [("dfs_analyzer", "dfs_analyzer")]
# include the formula module at root level
if os.path.exists("hypercube_formula_split.py"):
    datas += [("hypercube_formula_split.py", ".")]

# ── analysis ──────────────────────────────────────────────────────────────────
a = Analysis(
    ["run_gui.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["streamlit", "bokeh", "tornado", "IPython", "jupyter"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="RandomDFSAnalyzer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # no console window (windowed app)
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon="icon.ico",      # uncomment and set path if you have an icon
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="RandomDFSAnalyzer",
)
