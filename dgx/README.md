# Instructions to submit training job to Nvidia DGX cluster

## Quick Start Guide

SSH into the DGX cluster with the command below where `<$USER>` is your assigned User ID. This requires ASTAR VPN connection.
```
ssh <$USER>@10.218.13.21
```

From the login node, clone this repository into `DATA`. The end result should look like this: 
```
(base) <$USER>@dgx:~/DATA$ ls
rl-baselines3-zoo
```

## Building and running the Docker container

Modify the scripts `rl-baselines3-zoo/dgx/dgx_build_image.sh` and `rl-baselines3-zoo/dgx/dgx_run_image.sh` with your preferred settings. 
```
vim rl-baselines3-zoo/dgx/dgx_build_image.sh
```
Remember to change the email in the scripts to your I2R email. 
Refer to PowerPoint sent by Feri Guretno in email. 

Submit the job to build the Docker image to the `noGPU` queue.
```
qsub -q noGPU rl-baselines3-zoo/dgx/dgx_build_image.sh
```

Then submit the job to run the training within the Docker container. If you have configured the job to request GPU usage, then submit to a queue with GPUs like `lessGPU` or `lessGPU-long`.
```
qsub -q lessGPU-long rl-baselines3-zoo/dgx/dgx_run_image.sh 
```
Note that PBS Pro scheduler automatically logs stderr and stdout (in the format <script_name>.o<process_id>) to the location that the jobs were submitted from (i.e. `DATA`).

Check the status of your script with `qstat | grep $USER`. 

## Retrieving run results

Start an interactive session in the same node (e.g. dgx01 to dgx05) that you submitted your training job in.
```
qsub -I -q noGPU -l select=1:ngpus=0:ncpus=1:host=dgx02 -N isess -l software=nvidia-docker -l walltime=1:00:00 -m abe -M <your_email@i2r.a-star.edu.sg>
```

From the run log (<script_name>.o<process_id>), find the container ID or name that was used for your run. Alternatively you could also look at `docker ps -a`.
```
cat build_dgx_image.o22022 | grep INFO
```

Start an interactive bash session into the container. The container name should be of the form `sbl3-${GIT_COMMIT_SHA}`
```
docker exec -it <container_name> bash
```
If you see the following error, you are most likely not within an interactive session.
```
Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?
```

The internal filesystem of the Docker container contains the saved logs.
```
cd ~/rl-baselines3-zoo/logs/ppo
```

Copy out the training logs by first copying it from the Docker container to your `DATA` folder
```
docker cp <container_name>:/root/rl-baselines3-zoo/logs/ppo/ DATA/.
```

Then from a new terminal (without SSH) in the desired location, copy the logs to your desktop.
```
scp -r <$USER>@10.218.13.21:/home/i2r/<$USER>/DATA/ppo .
```



