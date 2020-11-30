#!/bin/bash
#SBATCH -J tfjob
#SBATCH -o tf_job%j.o
#SBATCH -t 05:00:00
#SBATCH -N 1 -n 28

module load Anaconda3
python VCA_2_1_MLP_RES_ES.py
