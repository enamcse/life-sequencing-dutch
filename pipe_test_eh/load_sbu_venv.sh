#!/bin/bash
# This is how the venv on regular snellius should be activated. For OSSC, change the ENV_NAME if needed.

declare ENV_NAME="myvenv"
# on OSSC
# declare ENV_NAME="/gpfs/ostor/ossc9424/homedir/virtual_envs/15oct2024/"

#Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv
