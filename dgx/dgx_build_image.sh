#!/bin/sh
#PBS -l select=1:ncpus=1:ngpus=0:host=dgx02
#PBS -l software=pytorch
#PBS -N build_dgx_image
#PBS -m abe
#PBS -M chuahmy@i2r.a-star.edu.sg

# For explanation of various settings above, see /apps/how-to/quick.reference01... in the DGX login node
RL_BASELINES_DIR="/home/i2r/$USER/DATA/rl-baselines3-zoo"

# First, cd to our git repository
# Next, build the docker image and save it in the docker registry
# This has to be done in the same command because it starts a subshell
(cd $RL_BASELINES_DIR && nvidia-docker build -t dgx-sbl3:latest \
	-f docker/Dockerfile-sbl3 .)