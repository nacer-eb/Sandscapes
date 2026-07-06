Code, figures and movies for the paper: *Sandscapes: self-modifying energy landscapes with emergent branching and flips*.

---

## Repository Structure

*   **`common/sandscapes_main.py`**: The core file used to simulate the model.
* **`FigureCode/`**: Code to generate specific figures, organized into subdirectories by figure number (e.g., `FigureCode/Fig1/`). This folder also includes a `generate_figures.sh` bash script to run the figure pipelines in the correct order.
*   **`Figures/`**: Saved image files of the final figures.
*   **`Movies/`**: Saved movies/animations of simulations, as referenced in the paper.
*   **`datasets/`**: This directory contains:
    *   Scripts to download, extract, and preprocess the full raw datasets.
    *   A preprocessed .npz used to generate Fig5.
