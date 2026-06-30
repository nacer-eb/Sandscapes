#!/bin/bash

# Define the target dataset
ACCESSION="GSE140802"
GEO_GROUP="GSE140nnn"

# Create a specific data subfolder if it doesn't exist and move into it
mkdir -p "${ACCESSION}_data"
cd "${ACCESSION}_data" || { echo "Failed to enter data directory"; exit 1; }

# Define the base FTP URLs using the variables
BASE_FTP="ftp://ftp.ncbi.nlm.nih.gov/geo/series/${GEO_GROUP}/${ACCESSION}"
SUPPL_DIR="${BASE_FTP}/suppl/"
MATRIX_URL="${BASE_FTP}/matrix/${ACCESSION}_series_matrix.txt.gz"
SOFT_URL="${BASE_FTP}/soft/${ACCESSION}_family.soft.gz"

echo "Fetching the list of supplementary files for ${ACCESSION} from NCBI GEO..."

# Dynamically fetch the supplementary file names and append the base directory to them
curl -s -l "$SUPPL_DIR" | while read -r filename; do
    # Strip carriage returns just in case to prevent malformed URLs
    filename=$(echo "$filename" | tr -d '\r')
    if [ -n "$filename" ]; then
        echo "${SUPPL_DIR}${filename}"
    fi
done > urls_to_download.txt

# Add the matrix and soft format files to the download list
echo "$MATRIX_URL" >> urls_to_download.txt
echo "$SOFT_URL" >> urls_to_download.txt

# Check if aria2c is installed; if not, fall back to wget
if command -v aria2c >/dev/null 2>&1; then
    echo "aria2c detected. Starting fast download..."
    aria2c -i urls_to_download.txt -j 5 -x 4
elif command -v wget >/dev/null 2>&1; then
    echo "aria2c not found. Falling back to wget..."
    wget -i urls_to_download.txt
else
    echo "Error: Neither aria2c nor wget is installed. Please install one of them."
    rm urls_to_download.txt
    exit 1
fi

# Clean up the temporary URL list
rm urls_to_download.txt

# Extract downloaded files
echo "Downloads complete. Starting extraction..."
for f in *.tar.gz *.tar; do
    if [ -f "$f" ]; then
        echo "Extracting $f..."
        if [[ "$f" == *.tar.gz ]]; then
            tar -xzvf "$f"
        elif [[ "$f" == *.tar ]]; then
            tar -xvf "$f"
        fi
    fi
done

echo "All ${ACCESSION} files downloaded and extracted successfully!"
