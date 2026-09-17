"""Online inference timing; filename deliberately avoids stdlib `profile`."""
import json
import platform
import statistics
import time

import torch
from torch import nn

import run as experiment

HERE, study, v = experiment.HERE, experiment.study, experiment.v


class Online(nn.Module):
    def __init__(self, calibrator, head):
        super().__init__()
        self.calibrator, self.head = calibrator, head

    def forward(self, leaves):
        raw = v.base.reduce_states(list(leaves.unbind(1)), v.base.motor_product, "balanced")
        cal = self.calibrator(leaves)
        return self.head(leaves, raw, cal)


@torch.inference_mode()
def main():
    if not (study.HERE / "results.json").exists():
        raise RuntimeError("wait until primary training campaign finishes before profiling")
    torch.set_num_threads(1)
    data = v.FoldData(v.ETH3DAll(), 0)
    checkpoints = torch.load(study.HERE / "runs/fold0_seed0.pt", weights_only=True)
    cal = v.CalibratedMotor()
    cal.load_state_dict(checkpoints["calibrator"])
    context = study.Head("contextual")
    context.load_state_dict(checkpoints["contextual"])
    mergeable = experiment.MergeableGate()
    mergeable.load_state_dict(torch.load(HERE / "runs/fold0_seed0.pt", weights_only=True))
    cached = study.cache(data, cal, "test")
    result = {"platform": platform.system() + " " + platform.machine(), "torch": torch.__version__,
              "threads": 1, "scope": "fold 0 seed 0 online end-to-end L32 prediction on recorded leaves; no tree parallelization", "models": {}}
    for name, head in (("contextual", context), ("mergeable", mergeable)):
        model = Online(cal, head).eval()
        difference = float((model(cached[0])-head(*cached[:3])).abs().max())
        assert difference <= 1e-6
        result["models"][name] = {"cached_online_max_difference": difference, "batches": {}}
        for batch in (1, 32, 256):
            leaves = cached[0][:batch]
            for _ in range(20):
                float(model(leaves)[0, 0])
            values = []
            for _ in range(100):
                started = time.perf_counter_ns()
                float(model(leaves)[0, 0])
                values.append((time.perf_counter_ns()-started)/1e6)
            median = statistics.median(values)
            result["models"][name]["batches"][str(batch)] = {"median_ms": median,
                "q1_ms": float(torch.tensor(values).quantile(.25)), "q3_ms": float(torch.tensor(values).quantile(.75)),
                "windows_per_second": batch*1000/median}
    study.save(HERE / "profile.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
