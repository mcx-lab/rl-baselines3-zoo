import os
import itertools

all_gait_names = ("walk", "trot", "canter", "pace")
all_gait_freqs = (1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9)
all_duty_factors = (0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8)

settings = itertools.chain(
    itertools.product(all_gait_names, all_gait_freqs, 0.75),
    itertools.product("walk", 1.5, all_duty_factors),
)

exp_id = 8

if __name__ == "__main__":
    for setting in settings:
        gait_name, gait_freq, duty_factor = setting
        cmd = f"""
            python scripts/enjoy_with_logging.py
            --algo ppo
            --env A1GymEnv-v0 
            -f logs/locomotion-baselines
            --exp-id {exp_id}
            --no-render
            --record
            --stats-dir {gait_name}-gaitfreq-{gait_freq}
            --env-kwargs
                gait_names:["'{gait_name}'",]
                gait_frequency_upper:{gait_freq}
                gait_frequency_lower:{gait_freq}
                duty_factor_lower:{duty_factor}
                duty_factor_upper:{duty_factor}
        """.replace(
            "\n", " "
        )
        print(cmd)
        os.system(cmd)
