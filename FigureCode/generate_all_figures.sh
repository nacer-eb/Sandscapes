#!/bin/bash

echo "Generating Figure 1..."
(cd Fig1 && bash generate_figure.sh)

echo "Generating Figure 2..."
(cd Fig2 && bash generate_figure.sh)

echo "Generating Figure 3..."
(cd Fig3 && bash generate_figure.sh)

echo "Generating Figure 4..."
(cd Fig4 && bash generate_figure.sh)

echo "Generating Figure 5..."
(cd Fig5 && bash generate_figure.sh)
