#!/bin/bash
RL_BASELINES_DIR=$1

# Collect all extra arguments to pass to training script
EXTRA_ARGS="${@:1}"

# First, cd to our git repository
# Next, build the docker image and save it in the docker registry
# This has to be done in the same command because it starts a subshell
(cd $RL_BASELINES_DIR && nvidia-docker build -t dgx-sbl3:latest \
	-f docker/Dockerfile-sbl3 .)

GIT_COMMIT_SHA=$(cd $RL_BASELINES_DIR && git rev-parse --short HEAD)
echo "INFO: Building container from commit ${GIT_COMMIT_SHA} in repository ${RL_BASELINES_DIR}..."
CONTAINER_NAME="sbl3-${GIT_COMMIT_SHA}"
echo "INFO: Using container name ${CONTAINER_NAME}..." 

# Delete the previous container if it exists
CONTAINER_ID=$(docker ps -aqf "name=${CONTAINER_NAME}")
if [[ $CONTAINER_ID != "" ]]
then
    echo "INFO: Reusing existing container ${CONTAINER_ID}..."
    nvidia-docker start $CONTAINER_ID

else
    echo "INFO: No pre-existing container found; creating new container..."
    # Create the container from the current image
    nvidia-docker run --privileged --net=host \
        --name=$CONTAINER_NAME \
        --env="DISPLAY=$DISPLAY" \
        --env="QT_X11_NO_MITSHM=1" \
        --runtime=nvidia \
        --gpus all \
        -dit \
        dgx-sbl3:latest \
        bash
fi

# Now we can run training scripts as usual. 
# PBS pro scheduler automatically logs stderr and stdout so no need to manually do that
CONTAINER_ID=$(docker ps -aqf "name=${CONTAINER_NAME}")
echo "INFO: Running train.py in container ${CONTAINER_ID}..."
CMD="cd ~/rl-baselines3-zoo && python train.py --algo ppo --env A1GymEnv-v0 --save-freq 100000 ${EXTRA_ARGS}"
echo "INFO: Running command ${CMD}"
nvidia-docker exec $CONTAINER_ID sh -c $CMD
