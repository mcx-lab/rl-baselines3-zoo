#!/bin/bash
RL_BASELINES_DIR=$1

# First, cd to our git repository
# Next, build the docker image and save it in the docker registry
# This has to be done in the same command because it starts a subshell
echo "Building container from ${RL_BASELINES_DIR}"
(cd $RL_BASELINES_DIR && nvidia-docker build -t dgx-sbl3:latest \
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
# PBS pro scheduler automatically logs stderr and stdout so no need to manually do that
CONTAINER_ID=$(docker ps -aqf "name=sbl3")
nvidia-docker exec $CONTAINER_ID sh -c "cd ~/rl-baselines3-zoo && python train.py --algo ppo --env A1GymEnv-v0 --save-freq 100000"
