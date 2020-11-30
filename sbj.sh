#!/bin/bash
#SBATCH -J tune2
#SBATCH -o tune2.o
#SBATCH -t 06:00:00
#SBATCH -N 1 -n 14

module load Anaconda3
python VCA_2_1_MLP_HP2.py
