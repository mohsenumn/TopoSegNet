#!/bin/bash
#SBATCH --job-name=topoUnet
#SBATCH --output=topoUnet.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --mem=100G

module load pytorch/2.2.0

time python -u main.py --params datalists/DRIVE/train_fields.json

