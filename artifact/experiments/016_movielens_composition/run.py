"""Evaluation-path composition on public, recorded MovieLens-1M ratings.

The task predicts the next recorded 1--5 star rating from a user's ordered
history and the queried movie. The data values and labels come only from the
official MovieLens-1M archive. Tree sampling changes computation scheduling;
it never creates, alters, or reorders an observation.

Development is the default. Passing --confirm is an explicit one-way action
that evaluates the frozen checkpoints on held-out users.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ARCHIVE = ROOT / "data/raw/ml-1m.zip"
EXPECTED_MD5 = "c4d9eecfca2ab87c1945afe126590906"
WIDTH = 9
HISTORY = 8
EVAL_LENGTHS = (4, 8, 16)
PATHS = ("left", "right", "balanced")
RANDOM_TREES = 16
ARMS = ("exact", "bilinear", "mlp", "mixed", "penalty")


@dataclass(frozen=True)
class Config:
    steps: int = 3000
    batch_size: int = 256
    lr: float = 2e-3
    weight_decay: float = 1e-4
    penalty_weight: float = 0.1
    eval_batch_size: int = 1024


def digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def user_split(user: int) -> str:
    bucket = int(hashlib.sha256(f"ml1m:{user}".encode()).hexdigest()[:8], 16) % 10
    if bucket < 8:
        return "train"
    return "dev" if bucket == 8 else "test"


class MovieLens:
    def __init__(self, archive: Path = ARCHIVE):
        if digest(archive, "md5") != EXPECTED_MD5:
            raise RuntimeError(f"MovieLens archive checksum mismatch: {archive}")
        raw: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
        movies: set[int] = set()
        with zipfile.ZipFile(archive) as zf, zf.open("ml-1m/ratings.dat") as fh:
            for row, line in enumerate(io.TextIOWrapper(fh, encoding="latin-1")):
                user_s, movie_s, rating_s, timestamp_s = line.rstrip().split("::")
                user, movie = int(user_s), int(movie_s)
                rating, timestamp = int(rating_s), int(timestamp_s)
                raw[user].append((timestamp, row, movie, rating))
                movies.add(movie)
        self.movie_to_index = {movie: i for i, movie in enumerate(sorted(movies))}
        self.sequences: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {
            "train": {}, "dev": {}, "test": {}
        }
        self.examples: dict[tuple[str, int], list[tuple[int, int]]] = {}
        for user, events in raw.items():
            events.sort()  # timestamp, then immutable source-file order
            movie = np.asarray([self.movie_to_index[e[2]] for e in events], dtype=np.int64)
            rating = np.asarray([e[3] for e in events], dtype=np.int64)
            self.sequences[user_split(user)][user] = (movie, rating)
        for split, users in self.sequences.items():
            for length in EVAL_LENGTHS:
                self.examples[(split, length)] = [
                    (user, endpoint)
                    for user, (movie, _rating) in users.items()
                    for endpoint in range(length, len(movie))
                ]
        self.train_examples = self.examples[("train", HISTORY)]
        self.n_ratings = sum(len(v) for users in self.sequences.values() for v, _ in users.values())

    @property
    def n_movies(self) -> int:
        return len(self.movie_to_index)

    def batch(self, examples: list[tuple[int, int]], length: int, indices) -> tuple[torch.Tensor, ...]:
        batch = [examples[int(i)] for i in indices]
        movies = np.empty((len(batch), length + 1), dtype=np.int64)
        ratings = np.empty((len(batch), length + 1), dtype=np.int64)
        labels = np.empty(len(batch), dtype=np.int64)
        users = np.empty(len(batch), dtype=np.int64)
        for j, (user, endpoint) in enumerate(batch):
            movie, rating = self.sequences[user_split(user)][user]
            movies[j] = movie[endpoint - length:endpoint + 1]
            ratings[j, :-1] = rating[endpoint - length:endpoint]
            ratings[j, -1] = 0  # mask the queried rating
            labels[j] = rating[endpoint] - 1
            users[j] = user
        return tuple(torch.from_numpy(x) for x in (movies, ratings, labels, users))

    def baselines(self, split: str, length: int) -> dict[str, dict[str, float]]:
        global_counts = Counter()
        movie_counts: dict[int, Counter] = defaultdict(Counter)
        for movie, rating in self.sequences["train"].values():
            for m, r in zip(movie, rating):
                global_counts[int(r)] += 1
                movie_counts[int(m)][int(r)] += 1
        global_mode = global_counts.most_common(1)[0][0]
        examples = self.examples[(split, length)]
        truth, global_pred, movie_pred, movie_nll = [], [], [], []
        alpha = 1.0
        for user, endpoint in examples:
            movie, rating = self.sequences[split][user]
            y, m = int(rating[endpoint]), int(movie[endpoint])
            counts = movie_counts.get(m, global_counts)
            truth.append(y)
            global_pred.append(global_mode)
            movie_pred.append(max(range(1, 6), key=lambda r: (counts[r], -r)))
            total = sum(counts.values()) + 5 * alpha
            movie_nll.append(-math.log((counts[y] + alpha) / total))
        return {
            "global_mode": classification_metrics(truth, global_pred),
            "movie_mode": classification_metrics(truth, movie_pred),
            "movie_distribution": {"nll": float(np.mean(movie_nll))},
        }


class Composer(nn.Module):
    def __init__(self, arm: str, n_movies: int):
        super().__init__()
        self.arm = arm
        self.movie = nn.Embedding(n_movies, WIDTH)
        self.rating = nn.Embedding(6, WIDTH)
        self.event_norm = nn.LayerNorm(WIDTH)
        self.final_norm = nn.LayerNorm(WIDTH)
        self.head = nn.Linear(WIDTH, 5)
        if arm != "exact":
            if arm == "mlp":
                self.op_net = nn.Sequential(
                    nn.Linear(2 * WIDTH, 25), nn.GELU(), nn.Linear(25, WIDTH),
                    nn.LayerNorm(WIDTH),
                )
            else:
                self.table = nn.Parameter(torch.empty(WIDTH, WIDTH, WIDTH))
                nn.init.normal_(self.table, std=1.0 / WIDTH)
                self.op_norm = nn.LayerNorm(WIDTH)

    def encode(self, movies: torch.Tensor, ratings: torch.Tensor) -> list[torch.Tensor]:
        z = self.movie(movies) + self.rating(ratings)
        if self.arm == "exact":
            eye = torch.eye(3, dtype=z.dtype, device=z.device).reshape(1, 1, WIDTH)
            z = eye + 0.05 * torch.tanh(z)
        else:
            z = self.event_norm(z)
        return list(z.unbind(dim=1))

    def op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.arm == "exact":
            shape = a.shape
            return (a.reshape(-1, 3, 3) @ b.reshape(-1, 3, 3)).reshape(shape)
        if self.arm == "mlp":
            return self.op_net(torch.cat((a, b), dim=-1))
        interaction = torch.einsum("bi,ijk,bj->bk", a, self.table, b) / math.sqrt(WIDTH)
        return self.op_norm(0.5 * (a + b) + interaction)

    def reduce(self, states: list[torch.Tensor], path: str, rng: random.Random | None = None):
        if len(states) == 1:
            return states[0]
        if path == "left":
            out = states[0]
            for state in states[1:]:
                out = self.op(out, state)
            return out
        if path == "right":
            out = states[-1]
            for state in reversed(states[:-1]):
                out = self.op(state, out)
            return out
        if path == "balanced":
            split = len(states) // 2
        elif path == "random":
            if rng is None:
                raise ValueError("random path requires an RNG")
            split = rng.randrange(1, len(states))
        else:
            raise ValueError(path)
        return self.op(self.reduce(states[:split], path, rng), self.reduce(states[split:], path, rng))

    def representation(self, movies, ratings, path="left", rng=None):
        return self.reduce(self.encode(movies, ratings), path, rng)

    def forward(self, movies, ratings, path="left", rng=None):
        return self.head(self.final_norm(self.representation(movies, ratings, path, rng)))


def classification_metrics(truth, pred) -> dict[str, float]:
    truth, pred = np.asarray(truth), np.asarray(pred)
    f1s = []
    for label in range(1, 6):
        tp = np.sum((truth == label) & (pred == label))
        fp = np.sum((truth != label) & (pred == label))
        fn = np.sum((truth == label) & (pred != label))
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    return {
        "accuracy": float(np.mean(truth == pred)),
        "macro_f1": float(np.mean(f1s)),
        "mae": float(np.mean(np.abs(truth - pred))),
    }


def composition_parameter_count(model: Composer) -> int:
    if model.arm == "exact":
        return 0
    names = ("op_net",) if model.arm == "mlp" else ("table", "op_norm")
    return sum(p.numel() for name, p in model.named_parameters() if name.startswith(names))


def train_one(data: MovieLens, arm: str, seed: int, cfg: Config, device: torch.device):
    torch.manual_seed(seed)
    rng = random.Random(seed + 20_000)
    gen = torch.Generator().manual_seed(seed + 10_000)
    model = Composer(arm, data.n_movies).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    model.train()
    started = time.perf_counter()
    moving = []
    for step in range(1, cfg.steps + 1):
        idx = torch.randint(len(data.train_examples), (cfg.batch_size,), generator=gen)
        movies, ratings, labels, _users = data.batch(data.train_examples, HISTORY, idx)
        movies, ratings, labels = movies.to(device), ratings.to(device), labels.to(device)
        if arm == "mixed":
            path = ("left", "right", "balanced", "random")[rng.randrange(4)]
            logits = model(movies, ratings, path, rng)
        else:
            logits = model(movies, ratings, "left")
        loss = nn.functional.cross_entropy(logits, labels)
        if arm == "penalty":
            states = model.encode(movies, ratings)
            start = rng.randrange(len(states) - 2)
            a, b, c = states[start:start + 3]
            lhs = model.op(model.op(a, b), c)
            rhs = model.op(a, model.op(b, c))
            penalty = ((lhs - rhs).square().mean())
            loss = loss + cfg.penalty_weight * penalty
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        moving.append(float(loss.detach()))
        if step % 500 == 0 or step == cfg.steps:
            print(f"{arm} seed={seed} step={step} loss={np.mean(moving[-100:]):.4f}", flush=True)
    return model, time.perf_counter() - started


@torch.no_grad()
def evaluate_path(model, data, split, length, path, cfg, device, tree_seed=0):
    examples = data.examples[(split, length)]
    all_truth, all_pred, all_nll, all_users, all_repr, all_logits = [], [], [], [], [], []
    rng = random.Random(tree_seed)
    for start in range(0, len(examples), cfg.eval_batch_size):
        idx = range(start, min(start + cfg.eval_batch_size, len(examples)))
        movies, ratings, labels, users = data.batch(examples, length, idx)
        movies, ratings = movies.to(device), ratings.to(device)
        representation = model.representation(movies, ratings, path, rng)
        logits = model.head(model.final_norm(representation))
        nll = nn.functional.cross_entropy(logits, labels.to(device), reduction="none")
        all_truth.extend((labels + 1).tolist())
        all_pred.extend((logits.argmax(-1).cpu() + 1).tolist())
        all_nll.extend(nll.cpu().tolist())
        all_users.extend(users.tolist())
        all_repr.append(representation.cpu())
        all_logits.append(logits.cpu())
    metrics = classification_metrics(all_truth, all_pred)
    metrics["nll"] = float(np.mean(all_nll))
    return (
        metrics, torch.cat(all_repr), torch.cat(all_logits), np.asarray(all_users),
        np.asarray(all_truth), np.asarray(all_pred),
    )


def user_clustered_delta_ci(base_correct, alt_correct, users, seed, draws=2000):
    """Cluster bootstrap preserving each resampled user's complete history."""
    unique = np.unique(users)
    delta = alt_correct - base_correct
    sums = np.asarray([delta[users == user].sum() for user in unique])
    counts = np.asarray([(users == user).sum() for user in unique])
    rng = np.random.default_rng(seed)
    boot = np.empty(draws)
    for draw in range(draws):
        sampled = rng.integers(0, len(unique), size=len(unique))
        boot[draw] = sums[sampled].sum() / counts[sampled].sum()
    return [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]


@torch.no_grad()
def evaluate(model, data, split, length, cfg, device, compute_ci=True):
    model.eval()
    started = time.perf_counter()
    output, representations, logits, predictions = {}, {}, {}, {}
    for path in PATHS:
        (output[path], representations[path], logits[path], users,
         truth, predictions[path]) = evaluate_path(model, data, split, length, path, cfg, device)
    random_rows, random_repr, random_logits, random_predictions = [], [], [], []
    for tree in range(RANDOM_TREES):
        row, rep, logit, _, _, pred = evaluate_path(
            model, data, split, length, "random", cfg, device, tree_seed=90_000 + tree
        )
        random_rows.append(row)
        random_repr.append(rep)
        random_logits.append(logit)
        random_predictions.append(pred)
    output["random_mean"] = {
        key: float(np.mean([row[key] for row in random_rows])) for key in random_rows[0]
    }
    base_rep, base_logits = representations["left"], logits["left"]
    base_correct = (predictions["left"] == truth).astype(np.float64)
    comparisons = {}
    for path in ("right", "balanced"):
        alt_correct = (predictions[path] == truth).astype(np.float64)
        comparisons[path] = {
            "paired_accuracy_delta": float(np.mean(alt_correct - base_correct)),
            "relative_representation_dispersion": float(
                ((representations[path] - base_rep).norm(dim=-1) /
                 base_rep.norm(dim=-1).clamp_min(1e-12)).mean()
            ),
            "max_abs_logit_difference": float((logits[path] - base_logits).abs().max()),
        }
        if compute_ci:
            comparisons[path]["user_clustered_95ci"] = user_clustered_delta_ci(
                base_correct, alt_correct, users, seed=71_000 + length
            )
    random_correct = np.mean([
        (pred == truth).astype(np.float64) for pred in random_predictions
    ], axis=0)
    comparisons["random_mean"] = {
        "paired_accuracy_delta": float(np.mean(random_correct - base_correct)),
        "relative_representation_dispersion": float(np.mean([
            ((rep - base_rep).norm(dim=-1) / base_rep.norm(dim=-1).clamp_min(1e-12)).mean()
            for rep in random_repr
        ])),
        "max_abs_logit_difference": float(max(
            (logit - base_logits).abs().max() for logit in random_logits
        )),
    }
    if compute_ci:
        comparisons["random_mean"]["user_clustered_95ci"] = user_clustered_delta_ci(
            base_correct, random_correct, users, seed=72_000 + length
        )
    return {
        "metrics": output,
        "path_comparisons": comparisons,
        "n_examples": len(users),
        "n_users": len(set(users.tolist())),
        "seconds": time.perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=Config.steps)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--confirm", action="store_true",
                        help="evaluate the frozen protocol on untouched test users")
    args = parser.parse_args()
    cfg = Config(steps=args.steps, batch_size=args.batch_size)
    device = torch.device(args.device)
    data = MovieLens()
    print(json.dumps({
        "dataset": "MovieLens-1M", "ratings": data.n_ratings,
        "movies": data.n_movies,
        "users": {k: len(v) for k, v in data.sequences.items()},
        "examples": {f"{s}_L{l}": len(data.examples[(s, l)])
                     for s in ("train", "dev", "test") for l in EVAL_LENGTHS},
    }, indent=2))
    out_dir = HERE / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    split = "test" if args.confirm else "dev"
    for arm in args.arms:
        for seed in args.seeds:
            checkpoint = out_dir / f"{arm}_seed{seed}.pt"
            model = Composer(arm, data.n_movies).to(device)
            train_seconds = 0.0
            if checkpoint.exists():
                model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
                print(f"loaded {checkpoint}")
            else:
                if args.confirm:
                    raise RuntimeError(f"confirmation requires frozen checkpoint: {checkpoint}")
                model, train_seconds = train_one(data, arm, seed, cfg, device)
                torch.save(model.state_dict(), checkpoint)
            row = {
                "arm": arm, "seed": seed, "split": split,
                "train_seconds": train_seconds,
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "composition_parameters": composition_parameter_count(model),
                "lengths": {},
            }
            for length in EVAL_LENGTHS:
                row["lengths"][str(length)] = evaluate(model, data, split, length, cfg, device)
                left = row["lengths"][str(length)]["metrics"]["left"]
                print(f"{arm} seed={seed} {split} L={length} "
                      f"acc={left['accuracy']:.4f} nll={left['nll']:.4f}", flush=True)
            if arm == "exact":
                model = model.to(device="cpu", dtype=torch.float64)
                row["float64_invariance_audit"] = {}
                for length in EVAL_LENGTHS:
                    audit = evaluate(
                        model, data, split, length, cfg, torch.device("cpu"), compute_ci=False
                    )
                    row["float64_invariance_audit"][str(length)] = {
                        path: values["max_abs_logit_difference"]
                        for path, values in audit["path_comparisons"].items()
                    }
            results.append(row)
    baselines = {str(length): data.baselines(split, length) for length in EVAL_LENGTHS}
    payload = {
        "protocol": "frozen-2026-09-03", "split": split,
        "config": asdict(cfg), "seeds": args.seeds, "arms": args.arms,
        "archive_md5": EXPECTED_MD5,
        "archive_sha256": digest(ARCHIVE, "sha256"),
        "baselines": baselines, "runs": results,
    }
    suffix = "confirmation" if args.confirm else "development"
    path = HERE / f"results_{suffix}.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
