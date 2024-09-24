"""
Python script to recreate the results of the IROS 2024 paper.
"""

import os

from platoon_gym.nn_train.train_lyapunov import DoubleIntLyapunovControllerTrainer
from platoon_gym.utils.utils import get_project_root

save_dir = os.path.join(get_project_root(), "scripts", "iros_2024")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


if __name__ == "__main__":
    trainer = DoubleIntLyapunovControllerTrainer(save_dir=save_dir)
    trainer.train_lyapunov_controller()
