#!/bin/bash

python3 A_simulate_weinreb.py
python3 A_FULL_TREE.py
python3 D_E_progenitor_plots.py
python3 F_progenitor_clusters_plots.py
python3 G_H_volcano_plots.py
python3 M_full_tree_movie.py

# Supplemental
python3 S_plot_markers.py
python3 S_WeinrebNCA.py
python3 S_plot_hist.py
