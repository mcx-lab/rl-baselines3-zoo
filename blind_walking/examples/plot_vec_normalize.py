import argparse
import pickle

from stable_baselines3.common.running_mean_std import RunningMeanStd


def repr_rms(rms: RunningMeanStd):
    return f"Mean: {rms.mean} -- Variance: {rms.var}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-path",
        help="The path to 'vecnormalize.pkl'. E.g. logs/ppo/A1GymEnv-v0_xx/A1GymEnv-v0/vecnormalize.pkl",
        type=str,
    )
    args = parser.parse_args()

    with open(args.save_path, "rb") as file_handler:
        vecnorm = pickle.load(file_handler)

    print(f"Observation clip: {vecnorm.clip_obs}")
    print(f"Reward clip: {vecnorm.clip_reward}")
    print("Reward statistics: ")
    print(repr_rms(vecnorm.ret_rms))
