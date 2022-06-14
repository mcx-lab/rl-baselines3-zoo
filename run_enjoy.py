import os
import itertools

# noise_levels = (0.01, 0.02, 0.05, 0.07)
seeds = (i for i in range(20))

# settings = itertools.product(noise_levels, seeds)

if __name__ == "__main__":

    # for setting in settings:
        # noise_level, seed = setting
    noise_level = 0.00
    for seed in seeds:
        command = f"""
        python scripts/enjoy_with_logging.py 
            --algo ppo 
            --env A1GymEnv-v0
            -f logs 
            --no-render 
            --env-kwargs
                terrain:"'step'"
            --stats-dir throw-obj-platform-seed{seed}
            --exp-id 44
            --seed {seed}
            --n-timesteps 3000
        """.replace("\n", " ").replace("\t", " ")
        print(command)
        os.system(command)