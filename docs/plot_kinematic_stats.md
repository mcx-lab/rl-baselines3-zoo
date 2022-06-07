# Plot kinematic stats

This document contains instructions on how to plot, for any policy: 
- upper / lower motor position
- max motor velocity
- max motor force 

## Quickstart 

First, use `scripts/enjoy_with_logging` in place of regular `enjoy`. 
All arguments are the same. In addition to playing the policy and optionally recording a video, it also saves various statistics to the checkpoint directory. 


Then,  open `plot_stats.ipynb`. 
You will need to configure `model_dir` to match the log folder and experiment ID. 
After that, run all cells. Everything should run without errors. 


