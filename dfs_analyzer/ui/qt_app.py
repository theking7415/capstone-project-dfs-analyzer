"""PyQt6 desktop GUI for Random DFS Graph Analyzer."""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Optional

import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter,
    QVBoxLayout, QHBoxLayout, QFormLayout, QStackedWidget,
    QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QProgressBar, QTextEdit, QTabWidget, QScrollArea,
    QSizePolicy, QMessageBox, QFileDialog, QLineEdit,
    QFrame,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QFontDatabase, QPalette, QColor

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

# Adds project root to import path
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Dark theme colour tokens
BG_DARK    = "#09090b"
BG_SURFACE = "#18181b"
BG_RAISED  = "#27272a"
BORDER_COL = "#3f3f46"
TEXT_PRI   = "#fafafa"
TEXT_SEC   = "#a1a1aa"
ACCENT     = "#6366f1"
ACCENT_HOV = "#818cf8"
ACCENT_DIM = "#4338ca"


# --- Background workers ---

class ExperimentWorker(QThread):
    """Runs a standard graph experiment in a background thread."""

    progress = pyqtSignal(int, int)
    finished = pyqtSignal(object)
    error    = pyqtSignal(str)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params

    def run(self):
        try:
            p = self.params
            from dfs_analyzer.experiments.config import ExperimentConfig
            from dfs_analyzer.experiments.runner import ExperimentRunner
            from dfs_analyzer.experiments.neighbor_runner import NeighborAnalysisRunner
            from dfs_analyzer.experiments.opposite_runner import OppositeAnalysisRunner
            from dfs_analyzer.experiments.custom_vertex_runner import CustomVertexRunner

            analysis = p["analysis_type"]
            cfg = ExperimentConfig(
                graph_type   = p["graph_type"],
                dimension    = p["dimension"],
                petersen_k   = p.get("petersen_k"),
                lattice_rows = p.get("lattice_rows"),
                lattice_cols = p.get("lattice_cols"),
                grid_size    = p.get("grid_size"),
                gnp_p        = p.get("gnp_p"),
                num_samples  = p["num_samples"],
                rng_seed     = p["rng_seed"],
                output_dir   = p["output_dir"],
                save_csv     = True,
            )

            cb = lambda cur, tot: self.progress.emit(cur, tot)

            if analysis == "Full Graph Analysis":
                results = ExperimentRunner().run(cfg, progress_callback=cb, num_processes=1)
            elif analysis == "Immediate Neighbors":
                results = NeighborAnalysisRunner().run(cfg, progress_callback=cb)
            elif analysis == "Opposite Vertex (Hypercube only)":
                results = OppositeAnalysisRunner().run(cfg, progress_callback=cb)
            elif analysis == "Custom Vertex Pair":
                results = CustomVertexRunner().run(
                    cfg,
                    start_vertex  = p["start_vertex"],
                    target_vertex = p["target_vertex"],
                    progress_callback=cb,
                )
            else:
                raise ValueError(f"Unknown analysis type: {analysis}")

            self.finished.emit(results)

        except Exception:
            self.error.emit(traceback.format_exc())


class ThreeVertexWorker(QThread):
    """Runs a three-vertex distribution experiment in a background thread."""

    progress = pyqtSignal(int, int)
    finished = pyqtSignal(object)
    error    = pyqtSignal(str)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params

    def run(self):
        try:
            p = self.params
            from dfs_analyzer.experiments.three_vertex_runner import ThreeVertexRunner
            cb = lambda cur, tot: self.progress.emit(cur, tot)
            result = ThreeVertexRunner().run(
                p["graph"], p["vertices"], p["labels"],
                n_samples=p["num_samples"],
                output_dir=p["output_dir"],
                graph_name=p["graph_name"],
                progress_callback=cb,
            )
            self.finished.emit(result)
        except Exception:
            self.error.emit(traceback.format_exc())


class AnalysisWorker(QThread):
    """Runs a post-experiment analysis tool in a background thread."""

    finished = pyqtSignal(object)
    error    = pyqtSignal(str)

    def __init__(self, tool: str, data_dir: str, graph_name: str):
        super().__init__()
        self.tool       = tool
        self.data_dir   = data_dir
        self.graph_name = graph_name

    def run(self):
        try:
            tool = self.tool
            dd   = self.data_dir
            gn   = self.graph_name

            if tool == "Deviation Analysis":
                from dfs_analyzer.core.deviation_analysis import run_deviation_analysis
                fig, result, layers, means, n = run_deviation_analysis(dd, graph_name=gn)
                self.finished.emit({"fig": fig, "result": result,
                                    "layers": layers, "means": means, "n": n})

            elif tool == "Sigmoid Model Fitting":
                from dfs_analyzer.core.sigmoid_fitting import run_sigmoid_fitting
                best, results, fig, layers, means, n = run_sigmoid_fitting(dd, graph_name=gn)
                self.finished.emit({"fig": fig, "best": best, "results": results, "n": n})

            elif tool == "Layer Variance Analysis":
                from dfs_analyzer.core.layer_variance import run_layer_variance_analysis
                result = run_layer_variance_analysis(dd, graph_name=gn)
                self.finished.emit(result)

            else:
                raise ValueError(f"Unknown tool: {tool}")

        except Exception:
            self.error.emit(traceback.format_exc())


# --- Reusable widgets ---

class FigureWidget(QWidget):
    """Embeds a matplotlib figure with a navigation toolbar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._canvas  = None
        self._toolbar = None

    def show_figure(self, fig):
        # Clears previous figure and renders the new one
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        canvas  = FigureCanvasQTAgg(fig)
        toolbar = NavigationToolbar2QT(canvas, self)
        self._layout.addWidget(toolbar)
        self._layout.addWidget(canvas)
        self._canvas  = canvas
        self._toolbar = toolbar
        canvas.draw()

    def clear(self):
        # Removes all child widgets from the layout
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()


class SectionLabel(QLabel):
    """Styled bold section heading label."""

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        f = self.font()
        f.setBold(True)
        f.setPointSize(9)
        self.setFont(f)
        self.setStyleSheet(f"color: {ACCENT}; padding-top: 8px; padding-bottom: 2px;")


# --- Graph type lists ---

GRAPH_TYPES = [
    "Hypercube",
    "Generalized Petersen",
    "Triangular Lattice",
    "Torus Grid",
    "Hexagonal Lattice",
    "Complete Graph",
    "N-Dimensional Grid",
    "G(n,p) Random",
    "Random d-Regular",
]

# Maps display name to internal config key
GRAPH_TYPE_MAP = {
    "Hypercube":            "hypercube",
    "Generalized Petersen": "petersen",
    "Triangular Lattice":   "triangular",
    "Torus Grid":           "torus",
    "Hexagonal Lattice":    "hexagonal",
    "Complete Graph":       "complete",
    "N-Dimensional Grid":   "ndgrid",
    "G(n,p) Random":        "gnp",
    "Random d-Regular":     "randomreg",
}


# --- Graph parameter stacked widget ---

class GraphParamStack(QStackedWidget):
    """Stacked widget with one parameter page per graph type."""

    paramsChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pages: dict[str, dict] = {}
        for gt in GRAPH_TYPES:
            self._add_page(gt)

    def _spin(self, lo, hi, val) -> QSpinBox:
        # Creates an integer spin box with the given range and default
        w = QSpinBox()
        w.setRange(lo, hi)
        w.setValue(val)
        w.valueChanged.connect(self.paramsChanged)
        return w

    def _dspin(self, lo, hi, val, dec=3) -> QDoubleSpinBox:
        # Creates a float spin box with the given range and default
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setValue(val)
        w.setDecimals(dec)
        w.setSingleStep(0.01)
        w.valueChanged.connect(self.paramsChanged)
        return w

    def _add_page(self, gt: str):
        # Builds and registers the parameter form for one graph type
        page    = QWidget()
        form    = QFormLayout(page)
        form.setSpacing(6)
        widgets = {}

        if gt == "Hypercube":
            d = self._spin(2, 20, 5)
            form.addRow("Dimension (d):", d)
            widgets["dimension"] = d

        elif gt == "Generalized Petersen":
            n = self._spin(3, 100, 5)
            k = self._spin(1, 49, 2)
            form.addRow("Ring size (n):", n)
            form.addRow("Skip parameter (k):", k)
            widgets["dimension"]  = n
            widgets["petersen_k"] = k
            # Clamps k to stay below n
            def clamp_k():
                k.setMaximum(max(1, n.value() - 1))
            n.valueChanged.connect(clamp_k)

        elif gt in ("Triangular Lattice", "Torus Grid", "Hexagonal Lattice"):
            r = self._spin(3, 200, 10)
            c = self._spin(3, 200, 10)
            form.addRow("Rows:", r)
            form.addRow("Columns:", c)
            widgets["lattice_rows"] = r
            widgets["lattice_cols"] = c

        elif gt == "Complete Graph":
            n = self._spin(2, 500, 10)
            form.addRow("Vertices (n):", n)
            widgets["dimension"] = n

        elif gt == "N-Dimensional Grid":
            d = self._spin(2, 10, 3)
            s = self._spin(2, 50, 5)
            form.addRow("Dimensions (d):", d)
            form.addRow("Size per dim:", s)
            widgets["dimension"] = d
            widgets["grid_size"] = s

        elif gt == "G(n,p) Random":
            n = self._spin(2, 5000, 50)
            p = self._dspin(0.001, 0.999, 0.3)
            form.addRow("Vertices (n):", n)
            form.addRow("Edge prob (p):", p)
            widgets["dimension"] = n
            widgets["gnp_p"]     = p

        elif gt == "Random d-Regular":
            n = self._spin(4, 5000, 100)
            d = self._spin(2, 20, 4)
            form.addRow("Vertices (n):", n)
            form.addRow("Degree (d):", d)
            widgets["dimension"]  = n
            widgets["petersen_k"] = d

        self._pages[gt] = widgets
        self.addWidget(page)

    def select(self, gt: str):
        # Switches the visible page to the given graph type
        self.setCurrentIndex(GRAPH_TYPES.index(gt))

    def get_params(self, gt: str) -> dict:
        # Returns current numeric values for the given graph type's widgets
        out: dict = {}
        for key, widget in self._pages[gt].items():
            if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                out[key] = widget.value()
        return out

    def get_vertex_count(self, gt: str) -> int:
        # Computes the number of vertices from current parameter values
        p = self.get_params(gt)
        if gt == "Hypercube":
            return 2 ** p.get("dimension", 5)
        elif gt == "Generalized Petersen":
            return 2 * p.get("dimension", 5)
        elif gt in ("Triangular Lattice", "Torus Grid", "Hexagonal Lattice"):
            return p.get("lattice_rows", 10) * p.get("lattice_cols", 10)
        elif gt == "Complete Graph":
            return p.get("dimension", 10)
        elif gt == "N-Dimensional Grid":
            return p.get("grid_size", 5) ** p.get("dimension", 3)
        elif gt in ("G(n,p) Random", "Random d-Regular"):
            return p.get("dimension", 50)
        return 0


# --- Three-vertex input panel ---

class ThreeVertexPanel(QWidget):
    """Input fields for specifying three vertices and their labels."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 4, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(SectionLabel("Three Vertices"))

        info = QLabel(
            "Enter three vertex identifiers.\n"
            "Hypercube: comma-separated bits.\n"
            "Lattices: row,col.  Petersen: ring,idx.\n"
            "Other graphs: integer index."
        )
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {TEXT_SEC}; font-size: 9pt;")
        lay.addWidget(info)

        form = QFormLayout()
        form.setSpacing(4)
        self.v1 = QLineEdit("0,0,0,0,0")
        self.v2 = QLineEdit("0,0,0,1,1")
        self.v3 = QLineEdit("1,1,1,1,1")
        self.l1 = QLineEdit("Near (L=1)")
        self.l2 = QLineEdit("Mid")
        self.l3 = QLineEdit("Far")
        form.addRow("Vertex 1:", self.v1)
        form.addRow("Label 1:",  self.l1)
        form.addRow("Vertex 2:", self.v2)
        form.addRow("Label 2:",  self.l2)
        form.addRow("Vertex 3:", self.v3)
        form.addRow("Label 3:",  self.l3)
        lay.addLayout(form)

    def get_inputs(self, gt: str) -> tuple[list, list]:
        # Parses vertex strings according to the graph type's vertex format
        raw  = [self.v1.text(), self.v2.text(), self.v3.text()]
        lbls = [self.l1.text(), self.l2.text(), self.l3.text()]
        verts = []
        for r in raw:
            r = r.strip()
            if gt == "Hypercube":
                verts.append(tuple(int(x.strip()) for x in r.split(",")))
            elif gt in ("Triangular Lattice", "Torus Grid", "Hexagonal Lattice"):
                parts = r.split(",")
                verts.append((int(parts[0].strip()), int(parts[1].strip())))
            elif gt == "Generalized Petersen":
                parts = r.split(",")
                verts.append((parts[0].strip(), int(parts[1].strip())))
            else:
                verts.append(int(r))
        return verts, lbls


# --- Custom vertex pair input panel ---

class CustomVertexPanel(QWidget):
    """Input fields for a custom start/target vertex pair."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 4, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(SectionLabel("Custom Vertex Pair"))
        info = QLabel(
            "Same format as Three-Vertex.\n"
            "Start vertex is the DFS origin.\n"
            "Target vertex is the focus."
        )
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {TEXT_SEC}; font-size: 9pt;")
        lay.addWidget(info)
        form = QFormLayout()
        form.setSpacing(4)
        self.sv = QLineEdit("0,0,0,0,0")
        self.tv = QLineEdit("1,1,1,1,1")
        form.addRow("Start vertex:", self.sv)
        form.addRow("Target vertex:", self.tv)
        lay.addLayout(form)

    def get_vertices(self, gt: str):
        # Parses and returns start and target vertex tuples
        def parse(r):
            r = r.strip()
            if gt == "Hypercube":
                return tuple(int(x.strip()) for x in r.split(","))
            elif gt in ("Triangular Lattice", "Torus Grid", "Hexagonal Lattice"):
                p = r.split(",")
                return (int(p[0].strip()), int(p[1].strip()))
            elif gt == "Generalized Petersen":
                p = r.split(",")
                return (p[0].strip(), int(p[1].strip()))
            else:
                return int(r)
        return parse(self.sv.text()), parse(self.tv.text())


# --- Left configuration panel ---

class ConfigPanel(QScrollArea):
    """Scrollable sidebar containing all experiment configuration controls."""

    run_requested = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setMinimumWidth(280)
        self.setMaximumWidth(380)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        inner = QWidget()
        self.setWidget(inner)
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)

        # Title label
        title = QLabel("Random DFS\nGraph Analyzer")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tf = title.font()
        tf.setBold(True)
        tf.setPointSize(13)
        title.setFont(tf)
        title.setStyleSheet(f"color: {TEXT_PRI}; padding: 10px 0 6px 0;")
        lay.addWidget(title)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"background: {BORDER_COL}; border: none; max-height: 1px;")
        lay.addWidget(sep)

        # Analysis type selector
        lay.addWidget(SectionLabel("1. Analysis Type"))
        self.analysis_combo = QComboBox()
        self.analysis_combo.addItems([
            "Full Graph Analysis",
            "Immediate Neighbors",
            "Opposite Vertex (Hypercube only)",
            "Custom Vertex Pair",
            "Three-Vertex Distribution",
        ])
        lay.addWidget(self.analysis_combo)

        # Graph type selector
        lay.addWidget(SectionLabel("2. Graph Type"))
        self.graph_combo = QComboBox()
        self.graph_combo.addItems(GRAPH_TYPES)
        lay.addWidget(self.graph_combo)

        # Graph parameter widgets (one page per graph type)
        lay.addWidget(SectionLabel("3. Graph Parameters"))
        self.param_stack = GraphParamStack()
        lay.addWidget(self.param_stack)

        # Live vertex count display
        self.vc_label = QLabel("Vertices: —")
        self.vc_label.setStyleSheet(f"color: {ACCENT_HOV}; font-size: 9pt;")
        lay.addWidget(self.vc_label)

        # Optional vertex input panels, hidden by default
        self.custom_panel = CustomVertexPanel()
        self.custom_panel.hide()
        lay.addWidget(self.custom_panel)

        self.three_panel = ThreeVertexPanel()
        self.three_panel.hide()
        lay.addWidget(self.three_panel)

        # Sampling controls
        lay.addWidget(SectionLabel("4. Sampling"))
        sform = QFormLayout()
        sform.setSpacing(4)
        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(10, 1_000_000)
        self.samples_spin.setValue(5000)
        self.samples_spin.setSingleStep(1000)
        sform.addRow("Samples:", self.samples_spin)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2**30)
        self.seed_spin.setValue(42)
        sform.addRow("RNG seed:", self.seed_spin)
        lay.addLayout(sform)

        # Output directory picker
        lay.addWidget(SectionLabel("5. Output Directory"))
        olay = QHBoxLayout()
        self.outdir_edit = QLineEdit("data_output")
        browse_btn = QPushButton("...")
        browse_btn.setFixedWidth(30)
        browse_btn.clicked.connect(self._browse_outdir)
        olay.addWidget(self.outdir_edit)
        olay.addWidget(browse_btn)
        lay.addLayout(olay)

        # Progress bar for sample count
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%  %v / %m")
        lay.addWidget(self.progress_bar)

        # Run button
        self.run_btn = QPushButton("Run Experiment")
        self.run_btn.setMinimumHeight(40)
        self.run_btn.setStyleSheet(
            f"QPushButton {{ background-color: {ACCENT}; color: {TEXT_PRI}; "
            "border: none; border-radius: 6px; font-size: 11pt; font-weight: bold; padding: 8px; }}"
            f"QPushButton:hover {{ background-color: {ACCENT_HOV}; }}"
            f"QPushButton:pressed {{ background-color: {ACCENT_DIM}; }}"
            f"QPushButton:disabled {{ background-color: {BG_RAISED}; color: {TEXT_SEC}; }}"
        )
        self.run_btn.clicked.connect(self._on_run)
        lay.addWidget(self.run_btn)

        lay.addStretch()

        # Version string
        ver = QLabel("v0.6.0  |  Ashoka University")
        ver.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ver.setStyleSheet(f"color: {TEXT_SEC}; font-size: 8pt;")
        lay.addWidget(ver)

        # Connect change signals
        self.graph_combo.currentTextChanged.connect(self._on_graph_changed)
        self.analysis_combo.currentTextChanged.connect(self._on_analysis_changed)
        self.param_stack.paramsChanged.connect(self._update_vc_label)

        # Set initial state
        self._on_graph_changed(self.graph_combo.currentText())

    def _on_graph_changed(self, gt: str):
        # Switches parameter page and refreshes vertex count
        self.param_stack.select(gt)
        self._update_vc_label()

    def _on_analysis_changed(self, at: str):
        # Shows or hides vertex input panels based on analysis type
        self.custom_panel.setVisible("Custom Vertex" in at)
        self.three_panel.setVisible("Three-Vertex" in at)
        if "Opposite" in at:
            # Opposite vertex requires hypercube; lock the graph selector
            self.graph_combo.setCurrentText("Hypercube")
            self.graph_combo.setEnabled(False)
        else:
            self.graph_combo.setEnabled(True)

    def _update_vc_label(self):
        # Recalculates and displays the vertex count for the current config
        gt = self.graph_combo.currentText()
        try:
            n = self.param_stack.get_vertex_count(gt)
            self.vc_label.setText(f"Vertices: {n:,}")
        except Exception:
            self.vc_label.setText("Vertices: —")

    def _browse_outdir(self):
        # Opens a folder picker and writes the chosen path to the output field
        d = QFileDialog.getExistingDirectory(self, "Select output directory",
                                             self.outdir_edit.text())
        if d:
            self.outdir_edit.setText(d)

    def _on_run(self):
        # Collects config, validates vertex inputs, then emits run_requested
        gt        = self.graph_combo.currentText()
        at        = self.analysis_combo.currentText()
        gtype_key = GRAPH_TYPE_MAP[gt]
        gparams   = self.param_stack.get_params(gt)

        params: dict = {
            "analysis_type": at,
            "graph_type":    gtype_key,
            "num_samples":   self.samples_spin.value(),
            "rng_seed":      self.seed_spin.value(),
            "output_dir":    self.outdir_edit.text(),
            **gparams,
        }

        # Lattice graphs expose rows/cols; set dimension for compatibility
        if "dimension" not in params and "lattice_rows" in params:
            params["dimension"] = params["lattice_rows"]

        if "Three-Vertex" in at:
            try:
                graph = _build_graph(gtype_key, gparams)
            except Exception as exc:
                QMessageBox.critical(self, "Graph error", str(exc))
                return
            vertices, labels = self.three_panel.get_inputs(gt)
            params["graph"]      = graph
            params["vertices"]   = vertices
            params["labels"]     = labels
            params["graph_name"] = gt

        elif "Custom Vertex" in at:
            try:
                sv, tv = self.custom_panel.get_vertices(gt)
            except Exception as exc:
                QMessageBox.critical(self, "Vertex parse error", str(exc))
                return
            params["start_vertex"]  = sv
            params["target_vertex"] = tv

        self.run_requested.emit(params)

    def set_running(self, running: bool):
        # Disables the run button and updates its label while an experiment runs
        self.run_btn.setEnabled(not running)
        self.run_btn.setText("Running..." if running else "Run Experiment")

    def set_progress(self, cur: int, tot: int):
        # Updates the progress bar to reflect current sample count
        if tot > 0:
            self.progress_bar.setMaximum(tot)
            self.progress_bar.setValue(cur)

    def reset_progress(self):
        # Resets the progress bar to zero
        self.progress_bar.setValue(0)


# --- Graph builder helper ---

def _build_graph(gtype_key: str, params: dict):
    # Instantiates and returns a graph object matching gtype_key
    from dfs_analyzer.core.graphs import (
        Hypercube, GeneralizedPetersen, TriangularLattice, TorusGrid,
        HexagonalLattice, CompleteGraph, NDGrid, RandomRegularGraph,
    )
    from dfs_analyzer.core.gnp_graph import generate_connected_gnp

    if gtype_key == "hypercube":
        return Hypercube(params["dimension"])
    elif gtype_key == "petersen":
        return GeneralizedPetersen(params["dimension"], params["petersen_k"])
    elif gtype_key == "triangular":
        return TriangularLattice(params["lattice_rows"], params["lattice_cols"])
    elif gtype_key == "torus":
        return TorusGrid(params["lattice_rows"], params["lattice_cols"])
    elif gtype_key == "hexagonal":
        return HexagonalLattice(params["lattice_rows"], params["lattice_cols"])
    elif gtype_key == "complete":
        return CompleteGraph(params["dimension"])
    elif gtype_key == "ndgrid":
        return NDGrid(params["dimension"], params["grid_size"])
    elif gtype_key == "gnp":
        return generate_connected_gnp(params["dimension"], params["gnp_p"])
    elif gtype_key == "randomreg":
        return RandomRegularGraph(n=params["dimension"], degree=params["petersen_k"])
    raise ValueError(f"Unknown graph type: {gtype_key}")


# --- Results panel (right side) ---

class ResultsPanel(QTabWidget):
    """Three-tab panel showing results, analysis tools, and documentation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTabPosition(QTabWidget.TabPosition.North)

        # Results tab
        self._exp_tab = QWidget()
        et_lay = QVBoxLayout(self._exp_tab)
        et_lay.setContentsMargins(6, 6, 6, 6)
        self._exp_figure = FigureWidget()
        self._exp_text   = QTextEdit()
        self._exp_text.setReadOnly(True)
        self._exp_text.setFont(QFont("Courier New", 9))
        self._exp_text.setMaximumHeight(180)
        self._exp_text.setPlaceholderText("Experiment summary will appear here...")
        et_lay.addWidget(self._exp_figure, stretch=3)
        et_lay.addWidget(self._exp_text,   stretch=1)
        self.addTab(self._exp_tab, "Results")

        # Analysis tools tab
        self._ana_tab = QWidget()
        self._build_analysis_tab()
        self.addTab(self._ana_tab, "Analysis Tools")

        # Documentation tab
        self._doc_tab = QWidget()
        doc_lay = QVBoxLayout(self._doc_tab)
        doc_edit = QTextEdit()
        doc_edit.setReadOnly(True)
        doc_edit.setMarkdown(_DOC_TEXT)
        doc_lay.addWidget(doc_edit)
        self.addTab(self._doc_tab, "Documentation")

    def _build_analysis_tab(self):
        # Populates the Analysis Tools tab with tool selector and run controls
        lay = QVBoxLayout(self._ana_tab)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        lay.addWidget(SectionLabel("Post-Experiment Analysis"))

        info = QLabel(
            "Select a completed experiment directory "
            "(must contain layer_statistics_bfs.csv)."
        )
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {TEXT_SEC}; font-size: 9pt;")
        lay.addWidget(info)

        form = QFormLayout()
        form.setSpacing(6)

        self.tool_combo = QComboBox()
        self.tool_combo.addItems([
            "Deviation Analysis",
            "Sigmoid Model Fitting",
            "Layer Variance Analysis",
        ])
        form.addRow("Tool:", self.tool_combo)

        dir_row = QHBoxLayout()
        self.ana_dir_edit = QLineEdit()
        self.ana_dir_edit.setPlaceholderText("Path to experiment directory...")
        browse_ana = QPushButton("...")
        browse_ana.setFixedWidth(30)
        browse_ana.clicked.connect(self._browse_ana_dir)
        dir_row.addWidget(self.ana_dir_edit)
        dir_row.addWidget(browse_ana)
        form.addRow("Experiment dir:", dir_row)

        self.ana_name_edit = QLineEdit()
        self.ana_name_edit.setPlaceholderText("Optional graph name for plot titles")
        form.addRow("Graph name:", self.ana_name_edit)

        lay.addLayout(form)

        self.ana_run_btn = QPushButton("Run Analysis")
        self.ana_run_btn.setMinimumHeight(36)
        self.ana_run_btn.setStyleSheet(
            f"QPushButton {{ background-color: {ACCENT}; color: {TEXT_PRI}; "
            "border: none; border-radius: 5px; font-weight: bold; padding: 6px; }}"
            f"QPushButton:hover {{ background-color: {ACCENT_HOV}; }}"
            f"QPushButton:pressed {{ background-color: {ACCENT_DIM}; }}"
            f"QPushButton:disabled {{ background-color: {BG_RAISED}; color: {TEXT_SEC}; }}"
        )
        self.ana_run_btn.clicked.connect(self._on_ana_run)
        lay.addWidget(self.ana_run_btn)

        self.ana_status = QLabel("")
        self.ana_status.setStyleSheet(f"color: {TEXT_SEC}; font-size: 9pt;")
        lay.addWidget(self.ana_status)

        self._ana_figure = FigureWidget()
        lay.addWidget(self._ana_figure, stretch=1)

        self._ana_worker: Optional[AnalysisWorker] = None

    def _browse_ana_dir(self):
        # Opens a folder picker and writes the path to the experiment dir field
        d = QFileDialog.getExistingDirectory(self, "Select experiment directory",
                                             self.ana_dir_edit.text())
        if d:
            self.ana_dir_edit.setText(d)

    def _on_ana_run(self):
        # Validates the experiment directory, then starts the analysis worker
        dd = self.ana_dir_edit.text().strip()
        if not dd or not Path(dd).is_dir():
            QMessageBox.warning(self, "Missing directory",
                                "Please select a valid experiment output directory.")
            return

        csv_path = Path(dd) / "layer_statistics_bfs.csv"
        if not csv_path.exists():
            QMessageBox.warning(
                self, "Missing CSV",
                f"No layer_statistics_bfs.csv found in:\n{dd}\n\n"
                "Run a Full Graph Analysis experiment first."
            )
            return

        tool = self.tool_combo.currentText()
        gn   = self.ana_name_edit.text().strip()

        self.ana_run_btn.setEnabled(False)
        self.ana_status.setText("Running...")

        self._ana_worker = AnalysisWorker(tool, dd, gn)
        self._ana_worker.finished.connect(self._on_ana_done)
        self._ana_worker.error.connect(self._on_ana_error)
        self._ana_worker.start()

    def _on_ana_done(self, result: dict):
        # Re-enables the run button and displays the result figure and summary
        self.ana_run_btn.setEnabled(True)
        tool = self.tool_combo.currentText()

        if tool == "Deviation Analysis":
            r   = result["result"]
            msg = (
                f"Deviation Analysis complete\n"
                f"R2 = {r.get('r_squared', 0):.4f}\n"
                f"a={r.get('a',0):.4f}  b={r.get('b',0):.4f}  "
                f"c={r.get('c',0):.4f}  d={r.get('d',0):.4f}"
            )
        elif tool == "Sigmoid Model Fitting":
            best = result.get("best", "-")
            res  = result.get("results", {})
            if best and best in res:
                msg = f"Sigmoid Fitting complete\nBest: {best}  R2 = {res[best].get('r2', 0):.4f}"
            else:
                msg = "Sigmoid Fitting complete (no successful fit)"
        elif tool == "Layer Variance Analysis":
            msg = "Layer Variance Analysis complete"
        else:
            msg = "Done"

        self.ana_status.setText(msg)
        fig = result.get("fig")
        if fig is not None:
            self._ana_figure.show_figure(fig)

    def _on_ana_error(self, tb: str):
        # Re-enables the run button and shows the error traceback
        self.ana_run_btn.setEnabled(True)
        self.ana_status.setText("Error - see details below")
        QMessageBox.critical(self, "Analysis error", tb)

    def show_experiment_results(self, results):
        # Displays the summary text and plot for a completed experiment
        plt.close("all")
        self.setCurrentIndex(0)
        try:
            summary = results.get_summary()
        except Exception:
            summary = str(results)
        self._exp_text.setPlainText(summary)
        try:
            fig = results.plot_figure()
            if fig is not None:
                self._exp_figure.show_figure(fig)
        except Exception as exc:
            self._exp_text.append(f"\n[Note: Could not generate plot - {exc}]")

    def show_three_vertex_results(self, result: dict):
        # Displays the three-vertex summary text and distribution figure
        plt.close("all")
        self.setCurrentIndex(0)
        ss    = result.get("summary_stats", {})
        lines = ["Three-Vertex Distribution - Summary\n"]
        for lbl, d in ss.items():
            lines.append(
                f"{lbl}: mean={d['mean']:.1f}  std={d['std']:.1f}  "
                f"min={d['min']}  max={d['max']}  "
                f"dev from (n-1)/2={d['deviation_from_half']:+.1f}"
            )
        self._exp_text.setPlainText("\n".join(lines))
        fig = result.get("fig")
        if fig is not None:
            self._exp_figure.show_figure(fig)


# --- Main window ---

class MainWindow(QMainWindow):
    """Top-level window containing the config panel and results panel."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Random DFS Graph Analyzer")
        self.resize(1280, 780)
        self.setMinimumSize(900, 600)
        self._worker: Optional[QThread] = None

        # Horizontal splitter: config on left, results on right
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        self.config_panel  = ConfigPanel()
        self.results_panel = ResultsPanel()

        splitter.addWidget(self.config_panel)
        splitter.addWidget(self.results_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([310, 970])

        self.statusBar().showMessage("Ready")

        self.config_panel.run_requested.connect(self._on_run_requested)

    def _on_run_requested(self, params: dict):
        # Blocks a second run while one is already in progress
        if self._worker and self._worker.isRunning():
            QMessageBox.warning(self, "Busy", "An experiment is already running.")
            return

        at = params.get("analysis_type", "")
        self.config_panel.set_running(True)
        self.config_panel.reset_progress()
        self.statusBar().showMessage("Running experiment...")

        if "Three-Vertex" in at:
            self._worker = ThreeVertexWorker(params)
            self._worker.finished.connect(self._on_three_vertex_done)
        else:
            self._worker = ExperimentWorker(params)
            self._worker.finished.connect(self._on_experiment_done)

        self._worker.progress.connect(self.config_panel.set_progress)
        self._worker.error.connect(self._on_experiment_error)
        self._worker.start()

    def _on_experiment_done(self, results):
        # Marks the run complete and forwards results to the results panel
        self.config_panel.set_running(False)
        self.config_panel.set_progress(
            self.config_panel.progress_bar.maximum(),
            self.config_panel.progress_bar.maximum(),
        )
        self.statusBar().showMessage("Experiment complete")
        self.results_panel.show_experiment_results(results)

    def _on_three_vertex_done(self, result: dict):
        # Marks the run complete and forwards three-vertex results to the results panel
        self.config_panel.set_running(False)
        self.config_panel.set_progress(
            self.config_panel.progress_bar.maximum(),
            self.config_panel.progress_bar.maximum(),
        )
        self.statusBar().showMessage("Three-vertex analysis complete")
        self.results_panel.show_three_vertex_results(result)

    def _on_experiment_error(self, tb: str):
        # Re-enables the panel and shows the error traceback in a dialog
        self.config_panel.set_running(False)
        self.statusBar().showMessage("Error - experiment failed")
        QMessageBox.critical(self, "Experiment failed", tb)


# --- Documentation text ---

_DOC_TEXT = """
# Random DFS Graph Analyzer - v0.6.0

**Research tool for validating the (n-1)/2 conjecture:**
the average discovery number of a vertex in randomised DFS on a large
symmetric regular graph tends to (n-1)/2.

---

## Analysis Types

| Type | Description |
|------|-------------|
| Full Graph Analysis | Mean discovery number for every vertex; layer statistics |
| Immediate Neighbors | Focus on the start vertex's immediate neighbors |
| Opposite Vertex | Vertex diagonally opposite in hypercube |
| Custom Vertex Pair | User-specified start and target vertices |
| Three-Vertex Distribution | Full histograms for three vertices at chosen BFS distances |

---

## Graph Types

| Graph | Vertices | Degree |
|-------|----------|--------|
| Hypercube d-D | 2^d | d |
| Generalized Petersen GP(n,k) | 2n | 3 |
| Triangular Lattice r x c | r * c | 6 |
| Torus Grid r x c | r * c | 4 |
| Hexagonal Lattice r x c | r * c | 3 |
| Complete Graph K_n | n | n-1 |
| N-Dimensional Grid | s^d | 2d |
| G(n,p) Random | n | ~(n-1)p |
| Random d-Regular | n | d |

---

## Post-Experiment Analysis Tools

These tools operate on a completed experiment directory
(must contain `layer_statistics_bfs.csv`):

- **Deviation Analysis** - fits a cubic polynomial to mean(L) - (n-1)/2
- **Sigmoid Model Fitting** - fits sigmoid A/(1+e^(-k*f(L))) with f=L, sqrt(L), log(L)
- **Layer Variance Analysis** - plots within-layer std dev and CV per BFS layer

---

## Workflow

1. Choose analysis type and graph type in the left panel
2. Set graph parameters, sample count, and output directory
3. Click **Run Experiment**
4. View results in the **Results** tab
5. For deeper analysis, switch to **Analysis Tools** and select the experiment directory

---

## Tips

- For the (n-1)/2 conjecture, use **Full Graph Analysis** with 5000+ samples
- Layer statistics are automatically saved to `layer_statistics_bfs.csv`
- **Three-Vertex Distribution** reveals how discovery order varies with BFS distance
- Deviation Analysis works best with 10000+ samples for stable polynomial fits

---

*Ashoka University - Capstone Project 2025*
*Author: Venkat Mahesh Mandava*
"""


# --- Stylesheet ---

_DARK_QSS = f"""
* {{
    font-family: "Helvetica Neue", "Helvetica", "Arial", sans-serif;
    font-size: 10pt;
    color: {TEXT_PRI};
    outline: none;
}}

QMainWindow, QDialog {{
    background: {BG_DARK};
}}

QWidget {{
    background: {BG_DARK};
    color: {TEXT_PRI};
}}

QSplitter::handle {{
    background: {BORDER_COL};
    width: 1px;
}}

QScrollArea {{
    background: {BG_SURFACE};
    border: none;
}}
QScrollArea > QWidget > QWidget {{
    background: {BG_SURFACE};
}}
QScrollBar:vertical {{
    background: {BG_SURFACE};
    width: 8px;
    border: none;
}}
QScrollBar::handle:vertical {{
    background: {BORDER_COL};
    border-radius: 4px;
    min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{
    background: {TEXT_SEC};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

QLabel {{
    background: transparent;
    color: {TEXT_PRI};
}}

QFrame[frameShape="4"],
QFrame[frameShape="5"] {{
    color: {BORDER_COL};
    background: {BORDER_COL};
}}

QGroupBox {{
    border: 1px solid {BORDER_COL};
    border-radius: 6px;
    margin-top: 8px;
    padding-top: 6px;
    color: {TEXT_SEC};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    color: {TEXT_SEC};
}}

QLineEdit {{
    background: {BG_RAISED};
    color: {TEXT_PRI};
    border: 1px solid {BORDER_COL};
    border-radius: 5px;
    padding: 4px 8px;
    selection-background-color: {ACCENT};
}}
QLineEdit:focus {{
    border: 1px solid {ACCENT};
}}
QLineEdit:disabled {{
    color: {TEXT_SEC};
    background: {BG_SURFACE};
}}

QSpinBox, QDoubleSpinBox {{
    background: {BG_RAISED};
    color: {TEXT_PRI};
    border: 1px solid {BORDER_COL};
    border-radius: 5px;
    padding: 4px 6px;
    selection-background-color: {ACCENT};
}}
QSpinBox:focus, QDoubleSpinBox:focus {{
    border: 1px solid {ACCENT};
}}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background: {BG_RAISED};
    border: none;
    width: 16px;
}}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background: {BORDER_COL};
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-bottom: 5px solid {TEXT_SEC};
    width: 0; height: 0;
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {TEXT_SEC};
    width: 0; height: 0;
}}

QComboBox {{
    background: {BG_RAISED};
    color: {TEXT_PRI};
    border: 1px solid {BORDER_COL};
    border-radius: 5px;
    padding: 5px 10px;
    selection-background-color: {ACCENT};
}}
QComboBox:focus {{
    border: 1px solid {ACCENT};
}}
QComboBox:hover {{
    border: 1px solid {TEXT_SEC};
}}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 24px;
    border: none;
    background: transparent;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid {TEXT_SEC};
    width: 0; height: 0;
    margin-right: 6px;
}}
QComboBox QAbstractItemView {{
    background: {BG_RAISED};
    color: {TEXT_PRI};
    border: 1px solid {BORDER_COL};
    selection-background-color: {ACCENT};
    selection-color: {TEXT_PRI};
    outline: none;
    padding: 2px;
}}
QComboBox QAbstractItemView::item {{
    min-height: 26px;
    padding: 2px 8px;
    color: {TEXT_PRI};
    background: transparent;
}}
QComboBox QAbstractItemView::item:hover {{
    background: {BORDER_COL};
    color: {TEXT_PRI};
}}
QComboBox QAbstractItemView::item:selected {{
    background: {ACCENT};
    color: {TEXT_PRI};
}}

QPushButton {{
    background: {BG_RAISED};
    color: {TEXT_PRI};
    border: 1px solid {BORDER_COL};
    border-radius: 5px;
    padding: 5px 12px;
}}
QPushButton:hover {{
    background: {BORDER_COL};
    border-color: {TEXT_SEC};
}}
QPushButton:pressed {{
    background: {BG_SURFACE};
}}
QPushButton:disabled {{
    color: {TEXT_SEC};
    background: {BG_SURFACE};
    border-color: {BG_RAISED};
}}

QProgressBar {{
    background: {BG_RAISED};
    color: {TEXT_PRI};
    border: 1px solid {BORDER_COL};
    border-radius: 5px;
    text-align: center;
    font-size: 8pt;
    height: 18px;
}}
QProgressBar::chunk {{
    background: {ACCENT};
    border-radius: 4px;
}}

QTextEdit {{
    background: {BG_RAISED};
    color: {TEXT_PRI};
    border: 1px solid {BORDER_COL};
    border-radius: 5px;
    padding: 4px;
    selection-background-color: {ACCENT};
}}
QTextEdit:focus {{
    border: 1px solid {ACCENT};
}}

QTabWidget::pane {{
    background: {BG_DARK};
    border: 1px solid {BORDER_COL};
    border-top: none;
    border-radius: 0 0 6px 6px;
}}
QTabBar {{
    background: {BG_DARK};
}}
QTabBar::tab {{
    background: {BG_SURFACE};
    color: {TEXT_SEC};
    border: 1px solid {BORDER_COL};
    border-bottom: none;
    padding: 7px 18px;
    margin-right: 2px;
    border-radius: 5px 5px 0 0;
    font-weight: bold;
}}
QTabBar::tab:selected {{
    background: {BG_RAISED};
    color: {TEXT_PRI};
    border-bottom: 2px solid {ACCENT};
}}
QTabBar::tab:hover:!selected {{
    background: {BG_RAISED};
    color: {TEXT_PRI};
}}

QStatusBar {{
    background: {BG_SURFACE};
    color: {TEXT_SEC};
    border-top: 1px solid {BORDER_COL};
    font-size: 8pt;
}}

QToolTip {{
    background: {BG_RAISED};
    color: {TEXT_PRI};
    border: 1px solid {BORDER_COL};
    padding: 4px 8px;
    border-radius: 4px;
}}

NavigationToolbar2QT {{
    background: {BG_SURFACE};
    border-bottom: 1px solid {BORDER_COL};
    spacing: 2px;
}}
NavigationToolbar2QT QToolButton {{
    background: transparent;
    border: none;
    padding: 4px;
    border-radius: 4px;
    color: {TEXT_PRI};
}}
NavigationToolbar2QT QToolButton:hover {{
    background: {BG_RAISED};
}}
"""


# --- Entry point ---

def main():
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName("Random DFS Graph Analyzer")
    app.setOrganizationName("Ashoka University")

    app.setStyle("Fusion")

    # Applies the dark Fusion palette
    pal = QPalette()
    c   = QColor
    pal.setColor(QPalette.ColorRole.Window,          c(BG_SURFACE))
    pal.setColor(QPalette.ColorRole.WindowText,      c(TEXT_PRI))
    pal.setColor(QPalette.ColorRole.Base,            c(BG_RAISED))
    pal.setColor(QPalette.ColorRole.AlternateBase,   c(BG_SURFACE))
    pal.setColor(QPalette.ColorRole.Text,            c(TEXT_PRI))
    pal.setColor(QPalette.ColorRole.ButtonText,      c(TEXT_PRI))
    pal.setColor(QPalette.ColorRole.Button,          c(BG_RAISED))
    pal.setColor(QPalette.ColorRole.Highlight,       c(ACCENT))
    pal.setColor(QPalette.ColorRole.HighlightedText, c(TEXT_PRI))
    pal.setColor(QPalette.ColorRole.ToolTipBase,     c(BG_RAISED))
    pal.setColor(QPalette.ColorRole.ToolTipText,     c(TEXT_PRI))
    pal.setColor(QPalette.ColorRole.PlaceholderText, c(TEXT_SEC))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, c(TEXT_SEC))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text,       c(TEXT_SEC))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, c(TEXT_SEC))
    app.setPalette(pal)

    # Sets font to Helvetica Neue with Arial fallback
    font = app.font()
    for name in ("Helvetica Neue", "Helvetica", "Arial"):
        font.setFamily(name)
        if QFont(name).exactMatch():
            break
    font.setPointSize(10)
    app.setFont(font)

    app.setStyleSheet(_DARK_QSS)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
