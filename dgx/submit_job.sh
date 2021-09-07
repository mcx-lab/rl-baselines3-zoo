#!/bin/bash
#PBS -l select=1:ncpus=1:ngpus=1:host=dgx02
#PBS -N rl_baselines3_zoo_train_a1_ppo
#PBS -l software=nvidia
#PBS -l walltime=6:00:00
#PBS -m abe
#PBS -M daniel_tan@i2r.a-star.edu.sg

# For explanation of various settings above, see /apps/how-to/quick.reference01... in the DGX login node
SCRIPT="${DATA}/rl-baselines3-zoo/dgx/train_a1_ppo.sh"
echo "Executing script ${SCRIPT}..."
bash $SCRIPT
