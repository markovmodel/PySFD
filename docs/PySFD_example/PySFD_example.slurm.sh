#!/bin/bash
# #SBATCH -J DR
#SBATCH -p big
# #SBATCH --nodes=1
# #SBATCH --array=0-6
# #SBATCH --cpus-per-task=64
#SBATCH --cpus-per-task=16
# #SBATCH --cpus-per-task=25
#SBATCH -o qlog.1.out
#SBATCH --time=30:00:00
# #SBATCH --mem=4000M
#SBATCH --mem=16000M
# #SBATCH --nodelist=gpu050

############
# Example script to execute PySFD on a supercomputer with
# a Slurm Workload Manager
############

# activate conda python3 environment
source activate python3

nohup python PySFD_example.py &> log
