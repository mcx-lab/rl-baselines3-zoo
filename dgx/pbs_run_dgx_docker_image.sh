#!/bin/sh
#PBS -l select=1:ncpus=12:ngpus=1:host=dgx02
#PBS -N rl-baselines3-zoo
#PBS -l walltime=12:00:00
#PBS -m abe
#PBS -M michael_chuah@i2r.a-star.edu.sg

nvidia-docker run -it --privileged --net=host \
     --name=sbl3 \
     --env="DISPLAY=$DISPLAY" \
     --env="QT_X11_NO_MITSHM=1" \
     --runtime=nvidia \
     --gpus all \
     dgx-sbl3:latest \
     bash

# Now the script should be inside the Docker container
# Print out some information for debugging
echo $USER
echo $(pwd)

# Cd to the root directory where rl-baselines3-repo is stored
cd
cd rl-baselines3-zoo

# Now we can run training scripts as usual.
# PBS pro scheduler automatically logs stderr and stdout so no need to manually do that
python train.py --algo ppo --env A1GymEnv-v0 --save-freq 100000