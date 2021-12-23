import rospy
from std_msgs.msg import String
import argparse
import os
import sys
import numpy as np
import yaml
import torch
import torchvision
from stable_baselines3 import PPO


def main():  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--log-path", help="Path to folder containing pre-trained model", type=str, default="./")
    args = parser.parse_args()

    env_id = "A1GymEnv-v0"
    log_path = args.log_path

    # ######################### Load model ######################### #

    # Check if we are running python 3.8+
    # we need to patch saved model under python 3.6/3.7 to load them
    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8
    custom_objects = {}
    if newer_python_version:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }

    model_path = os.path.join(log_path, f"{env_id}.zip")
    model = PPO.load(model_path, custom_objects=custom_objects, deterministic=True)
    print(f"Loaded model from {model_path}")

    # ######################### Trace model ######################### #

    policy = model.policy
    policy.eval()

    policy_feature_extractor = policy.features_extractor
    policy_mlp_extractor = policy.mlp_extractor
    policy_action = policy.action_net
    policy_net = policy.value_net
    sub_policies = [policy_feature_extractor, policy_mlp_extractor, policy_action, policy_net]
    sub_policies_names = ["feature_extractor", "mlp_extractor", "action_net", "value_net"]

    # trace model
    sub_policies_traced = [torch.jit.script(p) for p in sub_policies]

    # create directory if does not exist
    traced_log_path = os.path.join(log_path, "traced_model")
    if not os.path.exists(traced_log_path):
        os.mkdir(traced_log_path)

    # save traced model
    for i, tp in enumerate(sub_policies_traced):
        tp.save(os.path.join(traced_log_path, f"{sub_policies_names[i]}.pt"))
    print('Saved traced model')


if __name__ == "__main__":
    main()
