import os
import itertools

gait_names = ("walk", "trot")

gait_freqs = ((1.1, 1.9),)

duty_factors = ((0.55, 0.85),)

settings = itertools.product(gait_names, gait_freqs, duty_factors)

exp_id = 10

if __name__ == "__main__":
    for setting in settings:
        gait_name, gait_freq_range, duty_factor_range = setting
        cmd = f"""
            python enjoy.py
            --algo ppo
            --env A1GymEnv-v0 
            -f logs/locomotion_baselines
            --exp-id {exp_id}
            --no-render
            --record
        """.replace(
            "\n", " "
        )
        print(cmd)
        os.system(cmd)
