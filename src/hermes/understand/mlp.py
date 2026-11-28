"""A small torch MLP with real dropout, for the frustration detector.

Why torch rather than scikit-learn: `MLPClassifier` has no dropout. The
project brief asks for one hidden layer with dropout, weight decay and early
stopping, and sklearn can supply three of those four. torch is already a
dependency (CPU build), so this costs nothing new.

**It is an ensemble on purpose.** A single MLP on 180 French rows was measured
unstable to the point of uselessness: across five seeds it flagged 2, 8, 5, 8
and 10 of the 11 `correction` rows, while logistic regression flagged 5, 4, 5,
5, 5. Averaging the probabilities of several independently initialised nets
removes most of that variance, which is the only way an MLP here can be
compared with a linear model on equal terms.

Early stopping is deliberately **optional and off by default**. The sklearn
run held out 15 % of each training fold — about 21 rows — to decide when to
stop, and that 21-row signal is almost certainly what produced the seed
chaos. With this little data, fixed-length training under strong weight decay
is the more honest regulariser.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MLPConfig:
    """Everything the net needs. Defaults chosen for ~180 to ~1800 rows."""

    hidden: int = 48
    dropout: float = 0.3
    weight_decay: float = 1e-3
    learning_rate: float = 3e-3
    epochs: int = 200
    #: 128, not 32. With 144 French rows a batch of 32 is five steps an
    #: epoch, which is loop overhead rather than learning — the net has 18k
    #: parameters and a 32x384 matmul is microseconds of arithmetic.
    batch_size: int = 128
    #: Cap on gradient updates per member, which fixes a real confound in the
    #: **MLP arms only** and not only the runtime.
    #:
    #: At a fixed epoch count, a larger training set means more batches and
    #: therefore more updates: the French-only arms got 1,000 updates per net
    #: while the arms with 1600 EmoWOZ rows mixed in got 11,000. Those arms
    #: differed in two ways at once — extra data *and* eleven times the
    #: training — so a drop could be attributed to either. Capping updates
    #: holds training effort roughly constant, leaving the data as the only
    #: thing that varies.
    #:
    #: This says nothing about the logistic regression result. That model has
    #: no epochs and no batches — `max_iter` is a convergence limit, and it
    #: reaches the same optimum whatever the row count — so its drop from
    #: F1 0.93 to 0.81 under unweighted mixing is a pure data effect and must
    #: not be written up as partly over-training.
    max_updates: int = 800
    members: int = 5
    #: Hold out this fraction of the training rows to stop early. 0.0 disables
    #: it, which is the default — see the module docstring.
    validation_fraction: float = 0.0
    patience: int = 20


class TorchMLP:
    """Ensemble of one-hidden-layer nets, exposing a scikit-learn surface.

    Implements `fit`, `predict_proba`, `predict`, `get_params` and
    `set_params`, which is all `cross_val_predict` and `clone` require — so it
    drops into the same evaluation harness as the linear model instead of
    needing a parallel one.
    """

    def __init__(self, config: MLPConfig | None = None, seed: int = 0) -> None:
        self.config = config or MLPConfig()
        self.seed = seed
        self._members: list = []
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self.classes_ = np.array([0, 1])

    # -- the scikit-learn surface -----------------------------------------

    def get_params(self, deep: bool = True) -> dict:
        return {"config": self.config, "seed": self.seed}

    def set_params(self, **params) -> "TorchMLP":
        for key, value in params.items():
            setattr(self, key, value)
        return self

    # -- training ---------------------------------------------------------

    def _build(self, width: int, generator):
        import torch
        from torch import nn

        torch.manual_seed(int(generator.integers(0, 2**31 - 1)))
        return nn.Sequential(
            nn.Linear(width, self.config.hidden),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden, 1),
        )

    def _standardise(self, X):
        """Fit the scaler on these rows and return the scaled matrix."""
        self._mean = X.mean(axis=0)
        self._scale = X.std(axis=0) + 1e-6
        return (X - self._mean) / self._scale

    def _apply(self, X):
        return (X - self._mean) / self._scale

    def _loss_fn(self, y, weights):
        """Weighted BCE.

        Two weightings stack here and they answer different questions.
        `pos_weight` balances the classes, the torch equivalent of
        scikit-learn's `class_weight="balanced"`, so the two models stay
        comparable. `sample_weight` is per row, and is what lets a block of
        auxiliary training data carry the same *total* influence as the
        French rows rather than simply outnumbering them — 1600 English rows
        against 144 French ones took logistic regression from F1 0.93 to
        0.81, which is swamping, not transfer.
        """
        import torch
        from torch import nn

        positive = float(y.sum())
        negative = float(len(y) - positive)
        pos_weight = torch.tensor(
            [negative / positive if positive else 1.0], dtype=torch.float32
        )
        if weights is None:
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    def _epochs_for(self, rows: int, requested: int) -> int:
        """Epochs that keep total updates under `max_updates`."""
        import math

        per_epoch = max(1, math.ceil(rows / self.config.batch_size))
        if not self.config.max_updates:
            return requested
        return max(1, min(requested, math.ceil(self.config.max_updates / per_epoch)))

    def make_optimiser(self, net, lr: float):
        """An AdamW for this net. Exposed so a caller can keep one alive."""
        import torch

        return torch.optim.AdamW(
            net.parameters(), lr=lr, weight_decay=self.config.weight_decay,
        )

    def _run(self, net, Xs, y, weights, epochs, lr, rng, optimiser=None) -> None:
        """Train one net, capped at `max_updates` gradient steps.

        `optimiser` lets a caller train in segments on one optimiser. Building
        a fresh AdamW per segment would reset its moment estimates, so
        stopping at 50 epochs and continuing to 200 would not follow the same
        trajectory as training straight to 200 — which would make an epoch
        count chosen on the segmented run wrong for the real one.
        """
        import torch

        epochs = self._epochs_for(len(Xs), epochs)
        if optimiser is None:
            optimiser = self.make_optimiser(net, lr)
        loss_fn = self._loss_fn(y, weights)
        features = torch.from_numpy(Xs)
        labels = torch.from_numpy(y).unsqueeze(1)
        row_weights = (
            None if weights is None
            else torch.from_numpy(np.asarray(weights, dtype=np.float32)).unsqueeze(1)
        )

        for _epoch in range(epochs):
            net.train()
            order = torch.from_numpy(rng.permutation(len(features)))
            for start in range(0, len(features), self.config.batch_size):
                index = order[start : start + self.config.batch_size]
                optimiser.zero_grad()
                raw = loss_fn(net(features[index]), labels[index])
                loss = raw.mean() if row_weights is None else (
                    (raw * row_weights[index]).sum() / row_weights[index].sum()
                )
                loss.backward()
                optimiser.step()

    def fit(self, X, y, sample_weight=None) -> "TorchMLP":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        # Standardised on the training rows only. `fit` is called once per CV
        # fold, so the statistics never see the fold being tested.
        Xs = self._standardise(X)

        rng = np.random.default_rng(self.seed)
        self._members = []
        for _ in range(self.config.members):
            net = self._build(Xs.shape[1], rng)
            self._run(
                net, Xs, y, sample_weight,
                self.config.epochs, self.config.learning_rate, rng,
            )
            net.eval()
            self._members.append(net)
        return self

    def pretrain_then_finetune(
        self,
        X_pre,
        y_pre,
        X_fit,
        y_fit,
        finetune_lr: float | None = None,
        finetune_epochs: int | None = None,
    ) -> "TorchMLP":
        """Train on the auxiliary rows, then adapt on the target rows.

        A different bet from mixing the two sets: the auxiliary data shapes
        the representation, and the target data gets the last word on the
        decision boundary. Lower learning rate and few epochs on the second
        pass so the English prior is adapted rather than erased.

        The scaler is fitted on the **combined** rows, because the two passes
        must see the same feature space — refitting it between them would move
        the inputs under the net.
        """
        X_pre = np.asarray(X_pre, dtype=np.float32)
        y_pre = np.asarray(y_pre, dtype=np.float32).ravel()
        X_fit = np.asarray(X_fit, dtype=np.float32)
        y_fit = np.asarray(y_fit, dtype=np.float32).ravel()

        self._standardise(np.vstack([X_pre, X_fit]))
        pre_s, fit_s = self._apply(X_pre), self._apply(X_fit)

        lr = finetune_lr if finetune_lr is not None else self.config.learning_rate / 10
        epochs = (
            finetune_epochs if finetune_epochs is not None
            else max(10, self.config.epochs // 8)
        )

        rng = np.random.default_rng(self.seed)
        self._members = []
        for _ in range(self.config.members):
            net = self._build(pre_s.shape[1], rng)
            self._run(
                net, pre_s, y_pre,
                None, self.config.epochs, self.config.learning_rate, rng,
            )
            self._run(net, fit_s, y_fit, None, epochs, lr, rng)
            net.eval()
            self._members.append(net)
        return self

    # -- inference --------------------------------------------------------

    def predict_proba(self, X) -> np.ndarray:
        import torch

        if not self._members:
            raise RuntimeError("TorchMLP.predict_proba called before fit")

        X = np.asarray(X, dtype=np.float32)
        Xs = (X - self._mean) / self._scale
        tensor = torch.from_numpy(Xs)

        with torch.no_grad():
            probabilities = np.mean(
                [torch.sigmoid(net(tensor)).numpy().ravel() for net in self._members],
                axis=0,
            )
        return np.column_stack([1.0 - probabilities, probabilities])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    # -- persistence ------------------------------------------------------

    def save(self, path, metadata: dict | None = None) -> None:
        """Write the ensemble and its scaler, with whatever metadata matters.

        The scaler travels with the weights. It is fitted on the training
        rows, so a model loaded without it would standardise against the
        wrong statistics and score every utterance slightly wrong — the kind
        of failure that produces plausible numbers rather than an error.
        """
        import torch

        if not self._members:
            raise RuntimeError("nothing to save: fit the model first")

        torch.save(
            {
                "members": [net.state_dict() for net in self._members],
                "mean": self._mean,
                "scale": self._scale,
                "config": vars(self.config),
                "seed": self.seed,
                "metadata": metadata or {},
            },
            path,
        )

    @classmethod
    def load(cls, path, width: int) -> tuple["TorchMLP", dict]:
        """Rebuild an ensemble from disk. Returns the model and its metadata."""
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        config = MLPConfig(**payload["config"])
        model = cls(config, seed=int(payload.get("seed", 0)))
        model._mean = payload["mean"]
        model._scale = payload["scale"]

        rng = np.random.default_rng(model.seed)
        for state in payload["members"]:
            net = model._build(width, rng)
            net.load_state_dict(state)
            net.eval()
            model._members.append(net)
        return model, dict(payload.get("metadata", {}))


# --------------------------------------------------------------------------
# Choosing the epoch count instead of asserting it
# --------------------------------------------------------------------------

#: Epoch counts the inner selection chooses between. At 144 rows and batch
#: 128 that is 2 batches an epoch, so the candidates span 100 to 800 gradient
#: updates — and 400 epochs lands exactly on `max_updates`, so the cap never
#: silently truncates a candidate.
EPOCH_CANDIDATES: tuple[int, ...] = (50, 100, 200, 400)


def validation_curve(
    X,
    y,
    groups,
    config: MLPConfig,
    candidates: tuple[int, ...] = EPOCH_CANDIDATES,
    seed: int = 0,
    inner_splits: int = 3,
) -> dict[int, float]:
    """Mean inner-fold validation loss at each candidate epoch count.

    The split is `StratifiedGroupKFold` on `group`, for the same reason the
    outer split is: 42 % of the French rows are paraphrases of another row,
    and an inner split that separates two wordings of one complaint would
    choose the epoch count that best memorises templates.

    One net per inner fold, not an ensemble, and trained once to the largest
    candidate with the loss recorded as it passes each smaller one. Training
    separately per candidate would cost four times as much and measure the
    same thing.
    """
    import torch
    from sklearn.model_selection import StratifiedGroupKFold
    from torch import nn

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).ravel()
    groups = np.asarray(groups)

    # Fewer folds than asked for if the groups cannot support them.
    distinct = len(set(groups.tolist()))
    splits = max(2, min(inner_splits, distinct // 2))

    losses: dict[int, list[float]] = {c: [] for c in candidates}
    splitter = StratifiedGroupKFold(
        n_splits=splits, shuffle=True, random_state=seed,
    )

    for inner_train, inner_val in splitter.split(X, y.astype(int), groups):
        scratch = TorchMLP(config, seed=seed)
        mean = X[inner_train].mean(axis=0)
        scale = X[inner_train].std(axis=0) + 1e-6
        train_X = (X[inner_train] - mean) / scale
        val_X = torch.from_numpy((X[inner_val] - mean) / scale)
        val_y = torch.from_numpy(y[inner_val]).unsqueeze(1)

        rng = np.random.default_rng(seed)
        net = scratch._build(train_X.shape[1], rng)
        loss_fn = nn.BCEWithLogitsLoss()
        optimiser = scratch.make_optimiser(net, config.learning_rate)

        # Walk the candidates in order on one net and one optimiser, so the
        # loss at 200 is the loss of a net that trained straight to 200.
        trained = 0
        for candidate in sorted(candidates):
            scratch._run(
                net, train_X, y[inner_train], None,
                candidate - trained, config.learning_rate, rng,
                optimiser=optimiser,
            )
            net.train()
            trained = candidate
            net.eval()
            with torch.no_grad():
                losses[candidate].append(float(loss_fn(net(val_X), val_y)))

    return {c: float(np.mean(v)) for c, v in losses.items() if v}


def choose_epochs(
    X,
    y,
    groups,
    config: MLPConfig,
    candidates: tuple[int, ...] = EPOCH_CANDIDATES,
    seed: int = 0,
    inner_splits: int = 3,
) -> tuple[int, dict[int, float]]:
    """The epoch count with the lowest mean inner validation loss."""
    curve = validation_curve(
        X, y, groups, config, candidates, seed, inner_splits,
    )
    if not curve:
        return config.epochs, {}
    best = min(curve, key=lambda c: curve[c])
    return best, curve
