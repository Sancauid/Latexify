#!/bin/bash

set -e

rm -rf ./data
mkdir ./data

# Download the dataset into the directory
wget -O ./data/dataset.tgz https://storage.googleapis.com/i2l/data/dataset5.tgz

# Extract the dataset and set the correct permission
tar -xvzf ./data/dataset.tgz -C ./data
find ./data -type f -print0 | xargs -0 chmod 644  # For files
find ./data -type d -print0 | xargs -0 chmod 755  # For directories

# Remove the downloaded file
rm ./data/dataset.tgz
