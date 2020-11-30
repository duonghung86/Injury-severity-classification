#!/bin/bash
#SBATCH -J tfjob_auc_res
#SBATCH -o tf_job_auc_res.o
#SBATCH -t 06:00:00
#SBATCH -N 1 -n 14

module load Anaconda3
python VCA_2_MLP_AUC_RES.py
