# This file is slightly modified from train.py to send optimized_metric to vertex ai for hyperparameter tuning job
# can be run from any place

import hydra
import hypertune
import pyrootutils
from omegaconf import DictConfig

# project root setup
# searches for root indicators in parent dirs, like ".git", "pyproject.toml", etc.
# sets PROJECT_ROOT environment variable (used in `configs/paths/default.yaml`)
# loads environment variables from ".env" if exists
# adds root dir to the PYTHONPATH (so this file can be run from any place)
# https://github.com/ashleve/pyrootutils
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="train.yaml")
def main(cfg: DictConfig) -> float:

    # imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src.tasks.train_task import train
    from src.utils import get_metric_value

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag="metric", metric_value=metric_value
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
