#!/bin/bash
#PBS -l select=1:ncpus=1:host=dgx02
#PBS -N rl_baselines3_zoo_train_a1_ppo
#PBS -l software=nvidia
#PBS -l walltime=6:00:00
#PBS -m abe
#PBS -M daniel_tan@i2r.a-star.edu.sg

# For explanation of various settings above, see /apps/how-to/quick.reference01... in the DGX login node

# First, cd to our git repository. This assumes that we cloned the repo inside $DATA directory
# Next, build the docker image and save it in the docker registry
# This has to be done in the same command because it starts a subshell
(cd /home/i2r/daniel_tan/DATA && cd rl-baselines3-zoo && nvidia-docker build -t dgx-sbl3:latest \
	-f docker/Dockerfile-sbl3 .)

# Delete the previous container if it exists
CONTAINER_ID=$(docker ps -aqf "name=sbl3")
if [ $CONTAINER_ID != "" ] 
then
    echo "Deleting existing container ${CONTAINER_ID}..."
    docker rm -f $CONTAINER_ID
fi

# Create the container from the current image. This only works the first time
nvidia-docker run --privileged --net=host \
    --name=sbl3 \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --runtime=nvidia \
    --gpus all \
    -dit \
    dgx-sbl3:latest \
    bash

# Now we can run training scripts as usual. 
# PBS pro scheduleriautomatically logs stderr and stdout so no need to manually do that
CONTAINER_ID=$(docker ps -aqf "name=sbl3")
nvidia-docker exec $CONTAINER_ID sh -c "cd ~/rl-baselines-zoo && python train.py --algo ppo --env A1GymEnv-v0 --save-freq 100000"
