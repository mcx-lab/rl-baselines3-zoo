# Runtime profiling using cProfile

This directory supports an effort towards optimizing our codebase to reduce experiment runtimes. 

## How to profile

To profile a training experiment for 10k timesteps: 
```
python -m cProfile -o train_stats train.py ....<train.py args>.... -n 10000
```

## How to visualize stats:
```
python -m profile_stats.main --save-path train_stats
```
By default, this prints the stats for the top 10 most time-consuming functions. 
Advanced usage guide is available here: https://docs.python.org/3/library/profile.html

## List of stats:
- `blind_walking_stats`: generated from 3f283d0
- `terrain_aware_walking_stats`: generated from <TODO>
