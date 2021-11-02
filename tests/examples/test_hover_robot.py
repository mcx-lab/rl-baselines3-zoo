import subprocess

import pytest


@pytest.mark.slow
def test_hover_robot():
    completed_process = subprocess.run(["python", "blind_walking/examples/hover_robot.py", "--record"])
    completed_process.check_returncode()
