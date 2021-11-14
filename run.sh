#!/bin/bash
#SBATCH -p cuda10-ceshi
#SBATCH -N 1
#SBATCH --gres=gpu:4

python main.py
