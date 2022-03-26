from typing import Dict, Any
import json
import optuna
from pathlib import Path
from scipy import sparse
from loguru import logger
from functools import partial
from hydra_slayer import Registry
from torch_nlp_utils.common import Params
from src.first_level.models.core import Model
from torch_nlp_utils.common.params import with_fallback, parse_overrides


def fit_first_level(
    config: Dict[str, Any],
    train: Dict[str, sparse.csr_matrix],
    valid: Dict[str, sparse.csr_matrix] = None,
    hparams: Dict[str, Any] = None,
) -> Dict[str, Model]:
    registry = Registry(name_key="type")
    models = {}
    if hparams is None:
        if valid is None:
            raise ValueError("valid dataset is required for Optuna.")
        logger.debug("Hyper Parameters are not passed. Running optuna to find the best one.")
        hparams = run_optuna(config, train=train, valid=valid)
        logger.success("Finished optuna")
    for model in config["models"]:
        model_type = model.get("name") or model["model"]["type"].split(".")[-1]
        logger.info("Training {}", model_type)
        model_config = Params(
            with_fallback(
                preferred=(
                    parse_overrides(json.dumps(hparams[model_type]))
                    if model_type in hparams
                    else {}
                ),
                fallback=model,
            )
        )
        model_config = registry.get_from_params(**model_config.as_ordered_dict())
        metrics = model_config["model"].fit(
            train=train[model_type], valid=valid[model_type], config=model_config.get("train")
        )
        print(model_type, metrics)
        models[model_type] = model_config["model"]
    logger.success("Finished training all the first level models.")
    return models


def run_optuna(config: Dict[str, Any], train: Path, valid: Path) -> None:
    hparams = {}
    for idx, model_config in enumerate(config["models"]):
        if "optuna" not in model_config:
            continue
        optuna_config = model_config["optuna"]
        model_type = model_config.get("name") or model_config["model"]["type"].split(".")[-1]
        logger.info("Running optuna for {}", model_type)
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=3, n_warmup_steps=0, interval_steps=1
            ),
            study_name=f"first-level-{model_type}",
        )
        study.optimize(
            partial(
                objective,
                config=model_config,
                train=train[model_type],
                valid=valid[model_type],
                metric_key=optuna_config["metric"],
            ),
            **optuna_config["study"],
        )
        hparams[model_type] = study.best_params
    return hparams


def objective(
    trial: optuna.Trial,
    config: Dict[str, Any],
    train: Path,
    valid: Path,
    metric_key: str,
) -> Dict[str, Any]:
    registry = Registry(name_key="type")
    optuna_config = config["optuna"]
    params_overrides = {
        key: getattr(trial, value.get("suggest"))(
            name=key, **{k: v for k, v in value.items() if k != "suggest"}
        )
        for key, value in optuna_config["params"].items()
    }
    config = Params(
        with_fallback(preferred=parse_overrides(json.dumps(params_overrides)), fallback=config)
    )
    config = registry.get_from_params(**config.as_ordered_dict())
    metrics = config["model"].fit(train=train, valid=valid, config=config.get("train"))
    return metrics[metric_key]


if __name__ == "__main__":
    import yaml
    import numpy as np
    from scipy import sparse

    config_file = Path.cwd() / "configs" / "config.yaml"
    with config_file.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    train = {
        "ALS": sparse.csr_matrix(np.random.randint(low=0, high=2, size=(300, 300))),
        "MultVAE": (
            sparse.csr_matrix(np.random.randint(low=0, high=2, size=(300, 300))),
            sparse.csr_matrix(np.random.randint(low=0, high=2, size=(300, 300))),
        ),
        "EASE": sparse.csr_matrix(np.random.randint(low=0, high=2, size=(300, 300))),
        "EASE-check": sparse.csr_matrix(np.random.randint(low=0, high=2, size=(300, 300))),
        "SLIM": sparse.csr_matrix(np.random.randint(low=0, high=2, size=(300, 300))),
    }
    valid = {
        "ALS": sparse.csr_matrix(np.random.randint(low=0, high=2, size=(300, 300))),
        "MultVAE": (
            sparse.csr_matrix(np.random.randint(low=0, high=2, size=(300, 300))),
            sparse.csr_matrix(np.random.randint(low=0, high=2, size=(300, 300))),
        ),
        "EASE": sparse.csr_matrix(np.random.randint(low=0, high=2, size=(300, 300))),
        "EASE-check": sparse.csr_matrix(np.random.randint(low=0, high=2, size=(300, 300))),
        "SLIM": sparse.csr_matrix(np.random.randint(low=0, high=2, size=(300, 300))),
    }
    models = fit_first_level(config["first_level"], train=train, valid=valid)
    print(models)
