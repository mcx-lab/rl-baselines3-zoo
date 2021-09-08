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

4. Submit the job as follows:
```
qsub -q noGPU rl-baselines3-zoo/dgx/submit_job.sh
```
A suitable queue will be automatically assigned. 
To manually select a specific queue (e.g. noGPU), use `qsub -q noGPU <path/to/script>`. 

5. Check the status of your script with `qstat | grep $USER`. 

