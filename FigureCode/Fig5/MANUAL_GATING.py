import sys
import numpy as np

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.cm as cm

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vedo


#local
cam = dict(
    pos=(-95.1652, -40.9055, 32.9875),
    focal_point=(-2.27842, -4.03311, -4.58946),
    viewup=(0.355872, -0.933817, -0.0366230),
    roll=174.289,
    distance=106.769,
    clipping_range=(0.348629, 348.629),
)



matplotlib.rcParams.update({
    'font.size': 18,          # General text size
    'axes.titlesize': 24,     # Title size
    'axes.labelsize': 22,     # Axis label size (e.g., "Kit")
    'xtick.labelsize': 20,    # Tick number size
    'ytick.labelsize': 20,
    'legend.fontsize': 18
})

from MANUAL_GATING_plots import *



class AppWindow(QtWidgets.QMainWindow):
    def update_vedo(self, gate_mask):
        total_mask = self.master_mask & gate_mask
        self.update_base_colors(total_mask)
        final_colors_rgb_updated = np.copy(self.base_colors_rgb)
        final_colors_rgb_updated[~total_mask] = 0.0*final_colors_rgb_updated[~total_mask]
        self.cells.pointdata["rgb"] = np.copy(final_colors_rgb_updated)
        
        self.cells.mapper.SetScalarModeToUsePointFieldData()
        self.cells.mapper.SelectColorArray('rgb')
        self.cells.mapper.SetColorModeToDirectScalars()
        self.cells.mapper.SetScalarVisibility(True)
        self.fig.window.Render()

    def update_base_colors(self, total_mask):
        if self.current_color_mode == "CT_LABEL":
            self.base_colors_rgb = np.copy(self.pastel_colors[self.Ls_end])
        else:
            try:
                gene_id = np.argwhere(np.isin(np.char.upper(gene_names_high), np.char.upper(self.current_color_mode))).ravel()[0]
                expr = (Ms_saves_m1k @ pca_inv.T)[:, gene_id]
                p1, p99 = np.percentile(expr[total_mask], 1), np.percentile(expr[total_mask], 99)
                if p99 > p1:
                    norm_expr = np.clip((expr - p1) / (p99 - p1), 0, 1)
                else:
                    norm_expr = np.zeros_like(expr)
                self.base_colors_rgb = cm.viridis(norm_expr)[:, :3]
            except IndexError:
                return

    def on_color_changed(self, text):
        self.current_color_mode = text
        #self.update_base_colors()
        self.update_vedo(self.gate_masks.all(axis=0))
    
        
    def keyPressEvent(self, event):
        # If the user presses 'q' or 'Q', close the window
        if event.key() == Qt.Key_Q:
            self.close()

    def load_gate(self, gate_num):
        self.GATE = gate_num
        
        for id in range(len(self.axs)):
            if self.selector[id] != 0:
                self.selector[id].set_active(False)
                
            self.axs[id].clear()
            self.ids, self.scatter[id],\
                self.scatter_txt[id], self.selector[id] = create_mpl_plot(id, 
                                                                          self.axs[id],
                                                                          self.update_vedo,
                                                                          self.gate_masks,
                                                                          self.GATE, self.master_mask)
        self.canvas.draw()
        self.update_vedo(self.gate_masks.all(axis=0))
        
        
    def on_next_clicked(self):
        print("Next")
        self.master_mask = self.master_mask & self.gate_masks.all(axis=0)
        self.gate_masks = np.ones((len(self.axs), len(data_matrix_markers),), dtype=bool)
        self.load_gate(2)

    def on_reset_clicked(self):
        self.master_mask = np.ones(len(data_matrix_markers), dtype=bool)
        self.gate_masks = np.ones((len(self.axs), len(data_matrix_markers),), dtype=bool)
        self.load_gate(1)
        print("Reset")
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive Flow Gating - Prototype")
        
        # ----- Preparing the layout 
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.main_layout = QtWidgets.QHBoxLayout(self.central_widget)
        self.left_layout = QtWidgets.QVBoxLayout()
        self.button_layout = QtWidgets.QHBoxLayout()
        self.button_layout.setAlignment(Qt.AlignCenter)
        
        for text, callback in [("Next", self.on_next_clicked), ("Reset", self.on_reset_clicked)]:
            btn = QtWidgets.QPushButton(text)
            btn.setStyleSheet("font-size: 54px;")
            btn.clicked.connect(callback)
            self.button_layout.addWidget(btn)
        self.left_layout.addLayout(self.button_layout)

        # Color selection menu
        self.color_dropdown = QtWidgets.QComboBox()
        self.color_dropdown.setStyleSheet("font-size: 24px;")
        sorted_genes = sorted(list(gene_names_high))
        self.color_dropdown.addItem("CT_LABEL")
        self.color_dropdown.addItems(sorted_genes)
        self.color_dropdown.setEditable(True)
        self.color_dropdown.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        
        completer = QtWidgets.QCompleter(self.color_dropdown.model(), self.color_dropdown)
        completer.setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
        completer.setFilterMode(Qt.MatchContains)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.color_dropdown.setCompleter(completer)
        self.color_dropdown.currentTextChanged.connect(self.on_color_changed)      
        self.button_layout.addWidget(self.color_dropdown)
        
        
        self.fig_plt, self.axs_ = plt.subplots(2, 2, figsize=(8, 8))
        self.axs = self.axs_.ravel()
        self.canvas = FigureCanvas(self.fig_plt)
        
        self.left_layout.addWidget(self.canvas, 1)
        self.main_layout.addLayout(self.left_layout, 1)
        
        self.vtk_widget = QVTKRenderWindowInteractor(self.central_widget)
        self.main_layout.addWidget(self.vtk_widget, 1)
        # -----

        self.fig = vedo.Plotter(qt_widget=self.vtk_widget, axes=1)       
        self.cells, self.legend, self.Ls_end, self.selected_ct, self.pastel_colors  = create_vedo_plot()
        self.fig.show(self.cells, self.legend, camera=cam, axes=0)
        
        print(f"Number of cells {len(data_matrix_markers)}")
        self.master_mask = np.ones(len(data_matrix_markers), dtype=bool)
        self.gate_masks = np.ones((len(self.axs), len(data_matrix_markers),), dtype=bool)
        
        self.scatter = [0 for i in range(len(self.axs))]
        self.scatter_txt = [0 for i in range(len(self.axs))]
        self.selector = [0 for i in range(len(self.axs))]

        self.current_color_mode = "CT_LABEL"
        self.update_base_colors(self.master_mask)
        
        self.GATE = 1
        self.load_gate(self.GATE)
        

    def closeEvent(self, event):
        print("\n" + "="*50)
        print("FINAL GATING BOUNDARIES")
        print("="*50)
        
        # The specific gene pairs you are plotting
        for i in range(4):
            # Grab the extents from the Matplotlib RectangleSelector
            xmin, xmax, ymin, ymax = self.selector[i].extents
            
            # Lookup the annotated names
            x_name = matching_genes[marker_plotting_order][self.ids[i, 0]]
            y_name = matching_genes[marker_plotting_order][self.ids[i, 1]]
            
            # Print directly as Gene -> Bounds
            print(f"{x_name} bounds: [{xmin:.4f}, {xmax:.4f}]")
            print(f"{y_name} bounds: [{ymin:.4f}, {ymax:.4f}]")
            
        # Tell Qt it is safe to finish destroying the window
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = AppWindow()
    window.resize(3840, 2160)
    window.show()
    
    # Proper initialization for embedded VTK/Qt
    window.vtk_widget.Initialize()
    window.vtk_widget.Start()           # Use this instead of fig.interactor.Start()
    
    sys.exit(app.exec_())               # ← This is critical
