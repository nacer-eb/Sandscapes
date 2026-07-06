#!/bin/bash


(cd ../../datasets/mnist/ && python3 preprocess_data.py)

python3 B_C_simulate_triangle.py
python3 B_plot_3d_trajectory.py
python3 C_calc_triangle_energy.py
python3 C_energy_snapshots.py
python3 M_triangle_energy_movie.py

python3 D_E_simulate_mnist.py
python3 D_mnist_umap.py
python3 E_mnist_snapshots.py
python3 M_mnist_movie.py
