#!/bin/bash
#PBS -l select=1:ncpus=1:ngpus=0:host=dgx02
#PBS -N rl_baselines3_zoo_train_a1_ppo
#PBS -l software=nvidia
#PBS -l walltime=6:00:00
#PBS -m abe
#PBS -M daniel_tan@i2r.a-star.edu.sg

# For explanation of various settings above, see /apps/how-to/quick.reference01... in the DGX login node
RL_BASELINES_DIR="/home/i2r/daniel_tan/DATA/rl-baselines3-zoo"
SCRIPT="${RL_BASELINES_DIR}/dgx/train_a1_ppo.sh"
# Collect all command-line args and pass to training script
ARGS="$@"
echo "Executing script ${SCRIPT} from repository ${RL_BASELINES_DIR}..."
bash $SCRIPT $RL_BASELINES_DIR $ARGS
