#!/bin/bash
#SBATCH -J tfjob
#SBATCH -o tf_job%j.o
#SBATCH -t 02:00:00
#SBATCH -N 2 -n 28

module load Anaconda3
python VCA_4_1_MLP_optimization.py
