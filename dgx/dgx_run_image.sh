#!/bin/sh
#PBS -l select=1:ncpus=6:ngpus=2:host=dgx02
#PBS -l software=pytorch
#PBS -N run_dgx_image
#PBS -l walltime=12:00:00
#PBS -m abe
#PBS -M chuahmy@i2r.a-star.edu.sg

RL_BASELINES_DIR="/home/i2r/$USER/DATA/rl-baselines3-zoo"

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
# PBS Pro scheduler automatically logs stderr and stdout so no need to manually do that
CONTAINER_ID=$(docker ps -aqf "name=${CONTAINER_NAME}")

echo nvidia-smi

echo "INFO: Running train.py in container ${CONTAINER_ID}..."
nvidia-docker exec $CONTAINER_ID sh -c "cd ~/rl-baselines3-zoo && python train.py --algo ppo --env A1GymEnv-v0 --save-freq 100000"
