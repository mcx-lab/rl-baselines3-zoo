# Instructions to submit training job to DGX cluster

## Quick Start Guide

1. SSH into the DGX cluster. This requires ASTAR VPN connection
2. From the login node, clone this repository into $DATA. The end result should look like this: 
```
(base) daniel_tan@dgx:~/DATA$ ls
rl-baselines3-zoo
```
3. Modify the script `dgx/submit_job.sh` with your preferred settings. 

Refer to PowerPoint sent by Feri Guretno in email. 
Remember to change the email to your I2R email. 

4. Submit the job as follows. If you have configured the job to request GPUs in the previous step,
then submit to a queue with GPUs like lessGPU or lessGPU-long
```
qsub -q noGPU rl-baselines3-zoo/dgx/submit_job.sh
```
A suitable queue will be automatically assigned. 
To manually select a specific queue (e.g. noGPU), use `qsub -q noGPU <path/to/script>`. 

5. Check the status of your script with `qstat | grep $USER`. 

## Retrieving run results

1. Start an interactive session in the same node that you submitted your training job in. 
```
qsub -I -q noGPU select=1:ngpus=0:ncpus=1:host=dgx02 -N script1 -l software=nvidia-docker \ 
    -l walltime=1:00:00 -m abe -M <your_email@i2r.a-star.edu.sg>
```
2. From the run log (<script_name>.o<process_id>), find the container ID that was used for your run. 
```
cat rl_baselines_zoo_train_a1_ppo.o21984 | grep INFO
```
3. Start an interactive bash session into the container. 
```
docker exec -it <container_id> bash
```
4. The internal filesystem of the Docker container contains the saved logs.
```
cd ~/rl-baselines3-zoo/logs/ppo
```


