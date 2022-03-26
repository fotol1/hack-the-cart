from typing import List, Dict, Union, NamedTuple, Tuple, Iterable, Any
import math
import torch
import optuna
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy import sparse
from copy import deepcopy
from functools import partial
import torch.distributions as D
import torch.nn.functional as F
from functools import lru_cache
from catalyst import dl, metrics
from hydra_slayer import Registry
from torch.utils.data import DataLoader
from src.first_level.models.core import Model
from src.first_level.models.nn import FeedForward
from torch_nlp_utils.data import DatasetReader, CollateBatch, Batch

# -----------------------------------------------
# Model Definition
# -----------------------------------------------


class LatentSample(NamedTuple):
    z: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor


class KLScheduler:
    def __init__(
        self,
        zero_weight_steps: int,
        annealing_steps: int,
        max_weight: float = 1.0,
    ) -> None:
        self._step = 0
        self._weight = 0.0
        self._max_weight = max_weight
        self._zero_weight_steps = zero_weight_steps
        self._annealing_steps = annealing_steps

    @property
    def weight(self) -> Union[float, torch.Tensor]:
        return self._weight

    def step(self, state: Dict[str, torch.Tensor] = None) -> None:
        self._step += 1
        if not (self._zero_weight_steps > 0 and self._step <= self._zero_weight_steps):
            self._weight = min(
                self._max_weight, (self._step - self._zero_weight_steps) / self._annealing_steps
            )


class MultinomialLoss(torch.nn.Module):
    def __init__(self, size_average: bool = True) -> None:
        super().__init__()
        self._size_average = size_average

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None
    ) -> torch.Tensor:
        # logits, target, weights ~ (batch size, num classes)
        if weights is None:
            weights = torch.ones_like(logits)
        # Multiply with weights to discard padding
        log_probs = torch.log_softmax(logits, dim=-1) * weights
        # mult_loss ~ (batch size)
        mult_loss = -torch.einsum("bc,bc->b", log_probs, target)
        return mult_loss.mean() if self._size_average else mult_loss


class MultVAE(Model, torch.nn.Module):
    def __init__(
        self,
        encoder: FeedForward,
        decoder: FeedForward,
        kl_scheduler: KLScheduler,
        sample_size: int,
        normalize: bool = True,
        free_bits_alpha: float = 2.0,
        input_dropout: float = 0.5,
        metrics_topk: List[int] = [20, 100],
    ) -> None:
        super().__init__()
        self._encoder = encoder
        self._decoder = decoder
        self._kl_scheduler = kl_scheduler
        self._normalize = normalize
        if input_dropout is not None:
            self._input_dropout = torch.nn.Dropout(p=input_dropout)
        else:
            self._input_dropout = lambda x: x
        self._recon_error = MultinomialLoss()
        # Sampling
        self._free_bits_alpha = free_bits_alpha
        self._sample_size = sample_size
        self._mu_net = torch.nn.Linear(encoder.get_output_size(), sample_size)
        # self._logvar_net = torch.nn.Linear(encoder.get_output_size(), sample_size)
        self._sigma_net = torch.nn.Sequential(
            torch.nn.Linear(encoder.get_output_size(), sample_size),
            torch.nn.Softplus(),
            torch.nn.Hardtanh(min_val=1e-4, max_val=5.0),
        )
        # Fit predict
        self._metrics_topk = metrics_topk
        self._train_config = None

    @lru_cache(maxsize=None)
    def base_dist(self, device: torch.device = torch.device("cpu")) -> D.Normal:
        return D.Normal(
            loc=torch.zeros((self._sample_size,), device=device, dtype=torch.float),
            scale=torch.ones((self._sample_size,), device=device, dtype=torch.float),
        )

    def _encode(self, source: torch.Tensor) -> torch.Tensor:
        source = F.normalize(source, dim=-1) if self._normalize else source
        source = self._input_dropout(source)
        return self._encoder(source)

    def _decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self._decoder(latent)

    def _reparametrize(self, encoded: torch.Tensor, random: bool = True) -> LatentSample:
        # mu ~ (batch size, hidden size)
        self._mu = self._mu_net(encoded)
        # sigma ~ (batch size, hidden size)
        self._sigma = self._sigma_net(encoded)
        # sample ~ (batch size, samples, hidden size)
        sample = (
            self.base_dist(encoded.device).sample(self._mu.size()[:-1])
            if random
            else self._mu.new_zeros(self._mu.size())
        )
        z = self._mu + sample * self._sigma
        return LatentSample(z, self._mu, self._sigma)

    def _posterior_log_prob(self, latent: LatentSample) -> torch.Tensor:
        z, mu, sigma = latent
        log_prob = -0.5 * (
            (z - mu).pow(2) * sigma.pow(2).reciprocal() + 2 * sigma.log() + math.log(2 * math.pi)
        )
        # Sum over all dimensions except batch
        return torch.einsum("b...->b", log_prob)

    def _prior_log_prob(self, latent: LatentSample) -> torch.Tensor:
        return torch.einsum("b...->b", self.base_dist(latent.z.device).log_prob(latent.z))

    def forward(
        self, source: torch.Tensor, target: torch.Tensor = None, only_new_items: bool = False
    ) -> Dict[str, torch.Tensor]:
        # source ~ (batch size, num classes)
        # target ~ (batch size, num classes)
        encoded = self._encode(source)
        latent = self._reparametrize(encoded, random=self.training)
        logits = self._decode(latent.z)
        if self.training:
            self._kl_scheduler.step()
        if target is None:
            return {
                "source": source,
                "logits": logits,
                "probs": logits.softmax(dim=-1),
            }
        recon_error = self._recon_error(logits, target)
        # kl_loss = self._kl_loss(latent.mu, latent.logvar)
        posterior_log_prob = self._posterior_log_prob(latent)
        prior_log_prob = self._prior_log_prob(latent)
        kl_loss = (posterior_log_prob - prior_log_prob).clamp(min=self._free_bits_alpha)
        loss = recon_error + self._kl_scheduler.weight * kl_loss
        output_dict = {
            "source": source,
            "target": target,
            "logits": logits,
            "probs": logits.softmax(dim=-1),
            "loss": loss.mean(),
            "kl_weight": self._kl_scheduler.weight,
            "recon_error": recon_error.mean(),
            "kl_loss": kl_loss.mean(),
        }
        if not self.training and only_new_items:
            output_dict["logits"][source.gt(0)] = -1e13
        return output_dict

    def fit(
        self,
        config: Dict[str, Any],
        train: Tuple[sparse.csr_matrix, sparse.csr_matrix],
        valid: Tuple[sparse.csr_matrix, sparse.csr_matrix] = None,
    ) -> None:
        self._train_config = config
        loaders = build_dataloaders(config, train=train, valid=valid)
        runner = MultVAERunner()
        callbacks = {
            "tqdm": dl.TqdmCallback(),
            "backward": dl.BackwardCallback(metric_key="loss"),
            "optimizer": dl.OptimizerCallback(metric_key="loss"),
            "ndcg": dl.NDCGCallback(
                input_key="logits", target_key="target", topk=self._metrics_topk, log_on_batch=False
            ),
            # In Catalyst HitRate Metrics is actually Recall.
            "recall": dl.HitrateCallback(
                input_key="logits", target_key="target", topk=self._metrics_topk, log_on_batch=False
            ),
        }
        callbacks.update(config.get("callbacks", {}))
        runner.train(
            model=self,
            loaders=loaders,
            engine=config["engine"],
            optimizer=torch.optim.Adam(self.parameters(), lr=1e-3),
            num_epochs=config["epochs"],
            seed=13,
            timeit=False,
            callbacks=callbacks,
        )
        return runner.loader_metrics

    def predict(self, data: np.ndarray, remove_seen: bool = True) -> np.ndarray:
        loaders = build_dataloaders(
            self._train_config,
            test=(sparse.csr_matrix(data),),
        )
        runner = MultVAERunner()
        start = 0
        scores = np.zeros(data.shape)
        for output_dict in tqdm(
            runner.predict_loader(loader=loaders["test"], model=self),
            desc="Predict with MultVAE",
            total=len(loaders["test"]),
        ):
            end = output_dict["logits"].size(0)
            if remove_seen:
                output_dict["logits"][output_dict["source"].gt(0)] = -1e13
                output_dict["probs"] = output_dict["logits"].softmax(dim=-1)
            scores[start : start + end] = output_dict["probs"]
            start += end
        return scores


# -----------------------------------------------
# Training Definition
# -----------------------------------------------


class VAEDatasetReader(DatasetReader):
    """
    Dataset Reader for vae datasets.
    It accepts a directory which must contain `source.npz` and `target.npz` files
    created with `horec data prepare` command. These are sparse matrices
    that could be loaded with `load_npz`.

    Parameters
    ----------
    lazy : `bool`, optional (default = `False`)
        Whether to read dataset in lazy manner from the disk or not.
    filter_zeros : `bool`, optional (default = `True`)
        Whether to filter zeros in dataset or not.
        Setting this to False is useful if we are making predictions for batch of users.
    """

    def __init__(self, lazy: bool = False, filter_zeros: bool = True) -> None:
        # As we load an NPZ matrix in memory, lazy is not an option, max_instances_in_memory too
        super().__init__(lazy=lazy, max_instances_in_memory=None)
        self._filter_zeros = filter_zeros

    def _read(
        self, matrices: Tuple[sparse.csr_matrix, sparse.csr_matrix]
    ) -> Iterable[Dict[str, Any]]:
        # Source and target would have an equal number of rows
        # so it is ok to get an index like that
        source_matrix = matrices[0]
        target_matrix = matrices[1] if len(matrices) == 2 else None
        for idx in range(source_matrix.shape[0]):
            source = source_matrix[idx]
            target = target_matrix[idx] if target_matrix is not None else None
            # During training we do not need samples that consist from zeros only
            condition = (
                len(source.nonzero()[0]) == 0 or len(target.nonzero()[0]) == 0
                if target is not None
                else len(source.nonzero()[0]) == 0
            )
            if self._filter_zeros and condition:
                continue
            yield (
                {"source": source, "target": target} if target is not None else {"source": source}
            )


class VAECollateBatch(CollateBatch):
    def __init__(self, batch: Batch, only_new_items: bool = False) -> None:
        # Add float in the end just in case
        source = self._to_sparse_tensor(batch.source).to_dense().float()
        target = (
            self._to_sparse_tensor(batch.target).to_dense().float()
            if hasattr(batch, "target")
            else None
        )
        self.source, self.target = self._postprocess(source, target)
        if only_new_items:
            # Substitute target and source to get new items in target not present in source.
            substitute = self.target.gt(0).long() - self.source.gt(0).long()
            # Then we get a matrix with {-1, 0, 1}.
            # Where -1 - present in source but not in target.
            #        0 - not present/present in both.
            #        1 - present in target but not in source.
            # If we are computing performance only on new items, 1 is what we are loooking for.
            self.target = substitute.gt(0).float()
        self.only_new_items = only_new_items

    # Fix Catalyst batch size getter.
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.source

    @staticmethod
    def _to_sparse_tensor(rows: List[sparse.csr_matrix]) -> torch.sparse.Tensor:
        batch = sparse.vstack(rows)
        # Sparase matrix info
        samples, features = batch.shape
        values = batch.data
        coo_batch = batch.tocoo()
        indices = torch.LongTensor([coo_batch.row, coo_batch.col])
        sparse_tensor = torch.sparse.FloatTensor(
            indices, torch.from_numpy(values).float(), [samples, features]
        )
        return sparse_tensor

    @staticmethod
    def _postprocess(
        source: torch.Tensor, target: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            source.ne(0).float(),
            target.ne(0).float() if target is not None else None,
        )


class MultVAERunner(dl.Runner):
    def on_loader_start(self, runner: "MultVAERunner") -> None:
        super().on_loader_start(runner)
        self.meters = {
            key: metrics.AdditiveMetric(compute_on_call=False)
            for key in ("loss", "recon_error", "kl_loss", "kl_weight")
        }

    def handle_batch(self, batch: VAECollateBatch) -> None:
        batch = batch.to_device(self.engine.device, non_blocking=True)
        self.batch = self.model(**batch)
        for key in self.meters:
            metric = self.batch[key]
            self.meters[key].update(
                metric if isinstance(metric, float) else metric.item(),
                self.batch_size,
            )
            self.batch_metrics[key] = metric

    def predict_batch(self, batch: VAECollateBatch) -> Dict[str, torch.Tensor]:
        batch = batch.to_device(self.engine.device, non_blocking=True)
        return self.model(**batch)

    def on_loader_end(self, runner: "MultVAERunner") -> None:
        for key in self.meters:
            self.loader_metrics[key] = self.meters[key].compute()[0]
        super().on_loader_end(runner)


def multvae_collate_fn(instances: Iterable[Dict[str, List]], only_new_items: bool = False) -> Any:
    return VAECollateBatch(Batch(instances), only_new_items=only_new_items)


def build_dataloaders(
    config: Dict[str, Any], train: Path = None, valid: Path = None, test: Path = None
) -> Dict[str, DataLoader]:
    reader = VAEDatasetReader()
    loaders = {}
    if train is not None:
        loaders["train"] = DataLoader(
            reader.read(train), collate_fn=multvae_collate_fn, **config["loader"]
        )
    if valid is not None:
        loaders["valid"] = DataLoader(
            reader.read(valid),
            collate_fn=partial(multvae_collate_fn, only_new_items=True),
            **config["loader"],
        )
    if test is not None:
        config["loader"]["shuffle"] = False
        config["loader"]["drop_last"] = False
        loaders["test"] = DataLoader(
            reader.read(test), collate_fn=multvae_collate_fn, **config["loader"]
        )
    return loaders


if __name__ == "__main__":
    import json
    import tempfile
    import numpy as np
    from torch_nlp_utils.common import Params
    from torch_nlp_utils.common.params import with_fallback, parse_overrides

    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        registry = Registry(name_key="type")
        config = {
            "model": {
                "type": "src.first_level.models.multvae.MultVAE",
                "encoder": {
                    "type": "src.first_level.models.nn.FeedForward",
                    "input_size": 300,
                    "num_layers": 1,
                    "hidden_sizes": [600],
                    "activations": "tanh",
                },
                "decoder": {
                    "type": "src.first_level.models.nn.FeedForward",
                    "input_size": 200,
                    "num_layers": 2,
                    "hidden_sizes": [600, 300],
                    "activations": "tanh",
                },
                "kl_scheduler": {
                    "type": "src.first_level.models.multvae.KLScheduler",
                    "zero_weight_steps": 200,
                    "annealing_steps": 2000,
                    "max_weight": 0.2,
                },
                "input_dropout": 0.5,
                "sample_size": 200,
            },
            "train": {
                "engine": {"type": "catalyst.dl.CPUEngine"},
                "epochs": 10,
                "loader": {
                    "batch_size": 128,
                    "shuffle": True,
                    "drop_last": True,
                },
            },
            "optuna": {
                "model.encoder.hidden_sizes.0": {
                    "suggest": "suggest_int",
                    "low": 100,
                    "high": 200,
                },
                "model.sample_size": {
                    "suggest": "suggest_int",
                    "low": 100,
                    "high": 200,
                },
            },
        }
        with (temp / "config.json").open("w", encoding="utf-8") as file:
            json.dump(config, file, indent=2, ensure_ascii=False)
        train = sparse.csr_matrix(np.random.randint(low=0, high=2, size=(1000, 300)))
        valid = sparse.csr_matrix(np.random.randint(low=0, high=2, size=(1000, 300)))
        # User Params here with override variable for optuna
        config = Params.from_file(str(temp / "config.json"))
        if config.get("optuna") is not None:

            def objective(trial: optuna.Trial, config: Dict[str, Any]) -> float:
                params_overrides = {
                    key: getattr(trial, value.get("suggest"))(
                        name=key, **{k: v for k, v in value.items() if k != "suggest"}
                    )
                    for key, value in config["optuna"].items()
                }
                config = Params(
                    with_fallback(
                        preferred=parse_overrides(json.dumps(params_overrides)), fallback=config
                    )
                )
                config = registry.get_from_params(**config.as_ordered_dict())
                return 13

            study = optuna.create_study(
                direction="maximize",
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=1, n_warmup_steps=0, interval_steps=1
                ),
            )
            study.optimize(partial(objective, config=config), n_trials=3, timeout=300)
            print(study.best_params)
        config = Params.from_file(str(temp / "config.json"))
        config = registry.get_from_params(**config.as_ordered_dict())
        config["model"].fit(
            train=(train, deepcopy(train)), valid=(valid, deepcopy(valid)), config=config["train"]
        )
        print(config["model"].predict(valid.toarray(), remove_seen=True))
