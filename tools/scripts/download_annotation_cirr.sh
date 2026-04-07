#!/bin/bash

# Define colors
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create directory
mkdir -p /media/bd/PSSD/datasets/annotation/cirr

# Download Train annotations
echo -e "Downloading ${BLUE}CIRR Train${NC} annotations..."
wget https://raw.githubusercontent.com/Cuberick-Orion/CIRR/cirr_dataset/captions/cap.rc2.train.json -q -O /media/bd/PSSD/datasets/annotation/cirr/cap.rc2.train.json

# Download Val annotations
echo -e "Downloading ${BLUE}CIRR Val${NC} annotations..."
wget https://raw.githubusercontent.com/Cuberick-Orion/CIRR/cirr_dataset/captions/cap.rc2.val.json -q -O /media/bd/PSSD/datasets/annotation/cirr/cap.rc2.val.json

# Download Test annotations
echo -e "Downloading ${BLUE}CIRR Test${NC} annotations..."
wget https://raw.githubusercontent.com/Cuberick-Orion/CIRR/cirr_dataset/captions/cap.rc2.test1.json -q -O /media/bd/PSSD/datasets/annotation/cirr/cap.rc2.test1.json
