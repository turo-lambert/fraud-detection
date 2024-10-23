#!/bin/bash -e

#TODO: Replace "fraud-detection" by the name of your repo, or any name you
# want to give to your env.
read -p "Want to install conda env named 'fraud-detection'? (y/n)" answer
if [ "$answer" = "y" ]; then
  echo "Installing conda env..."
  conda create -n fraud-detection python=3.10 -y
  source $(conda info --base)/etc/profile.d/conda.sh
  conda activate fraud-detection
  echo "Installing requirements..."
  pip install -r requirements-developer.txt
  python3 -m ipykernel install --user --name=fraud-detection
  conda install -c conda-forge --name fraud-detection notebook -y
  echo "Installing pre-commit..."
  make install_precommit
  echo "Installation complete!";
else
  echo "Installation of conda env aborted!";
fi
