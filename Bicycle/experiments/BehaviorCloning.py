import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.JointTraining import JointTrainingConfig, run_joint_training


def main() -> None:
    run_joint_training(
        JointTrainingConfig(
            behavior_cloning_only=True,
            results_dir="results/BehaviorCloning",
            pretrained_controller_path="",
            freeze_pretrained_controller=False,
        )
    )


if __name__ == "__main__":
    main()
