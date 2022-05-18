import os
import itertools

# TARGET_VELS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
TARGET_VELS = (0.5,)

GAITS = ("walk", "trot", "canter", "pace")
DFS = (0.25, 0.5, 0.75)
GAIT_FREQS = (0.5, 1.5, 2.5)

settings = itertools.product(GAITS, GAIT_FREQS, DFS)

if __name__ == "__main__":
    for gait_name, gait_freq, df in settings:
        cmd = f"""
            python scripts/enjoy_with_logging.py 
            --algo ppo 
            --env A1GymEnv-v0 
            -f logs/new-locomotion-baselines
            --no-render 
            --record
            --env-kwargs
                gait_names:["'{gait_name}'"]
                gait_frequency_upper:{gait_freq}
                gait_frequency_lower:{gait_freq}
                duty_factor_upper:{df}
                duty_factor_lower:{df}
            --stats-dir {gait_name}-gf{gait_freq}-df{df}
        """.replace(
            "\n", ""
        )
        os.system(cmd)
