#!/bin/bash
# #SBATCH -J DR
#SBATCH -p big
# #SBATCH --nodes=1
# #SBATCH --array=0-3
#SBATCH --array=0-6

#SBATCH --cpus-per-task=5

#SBATCH -o qlogs/%j.out
#SBATCH --time=30:00:00

#SBATCH --mem=25000M

# #SBATCH --nodelist=gpu050
# #SBATCH --exclude=cmp247

############
# Example script to execute PySFD on a supercomputer with
# a Slurm Workload Manager
############

# activate conda python3 environment
source activate python3

mkdir -p logs

l_num_bs=(25 50 75 100 125 150 162)
num_bs=${l_num_bs[$SLURM_ARRAY_TASK_ID]}
for runID in $(seq 0 1 100)
do
#FeatureTypeInd=0
FeatureTypeInd=1
nohup python PySFD_example.convcheck.py $FeatureTypeInd $num_bs $runID &> logs/log.$num_bs.$runID.$FeatureTypeInd
done
