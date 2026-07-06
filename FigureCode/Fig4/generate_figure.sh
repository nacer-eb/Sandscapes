#!/bin/bash

python3 A_calc_catastrophe.py
python3 A_catastrophe_plot_growth.py
python3 B_catastrophe_trajectory_growth_plot.py

# This below is slightly inefficient (The memmap is heavy, but only used in C)
python3 B_simulate_triangle_growth.py
python3 B_dynamical_plots.py
python3 C_dynamical_trajectories.py
## -- ##

python3 D_E_simulate_5Xis.py
python3 D_E_calc_5Xis_energy.py
python3 D_E_5Xis_energy_movie_n_snapshots.py
