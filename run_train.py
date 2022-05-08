import os
import itertools

gait_namess = (
    "[\"'pace'\"]",
    "[\"'canter'\"]",
    "[\"'walk'\",\"'trot'\"]",
)

gait_freqs = ((1.1, 1.9),)

duty_factors = (
    (0.55, 0.85),
    # (0.75, 0.75)
)


def get_gait_names_str(gait_names):
    return '""'


settings = itertools.product(gait_namess, gait_freqs, duty_factors)

if __name__ == "__main__":
    for setting in settings:
        gait_names, gait_freq_range, duty_factor_range = setting
        cmd = f"""
            python train.py
            --algo ppo
            --env A1GymEnv-v0 
            -f logs/locomotion-baselines
            --n-timesteps 2000000
            --save-freq 100000
            --env-kwargs
                gait_names:{gait_names}
                gait_frequency_upper:{gait_freq_range[0]}
                gait_frequency_lower:{gait_freq_range[1]}
                duty_factor_upper:{duty_factor_range[0]}
                duty_factor_lower:{duty_factor_range[1]}
                obs_steps_ahead:[0,1,2,4,8]
            --use-wandb 
            --project-name locomotion_baselines
            --run-name {gait_names}-obs-01248-gaitfreq-{gait_freq_range[0]}-{gait_freq_range[1]}-duty-{duty_factor_range[0]}-{duty_factor_range[1]}
            --tensorboard tensorboard_log
        """.replace(
            "\n", " "
        )
        print(cmd)
        os.system(cmd)
