"""Check cached versus original online contextual computation on real inputs."""
import json

import torch

import run as study


@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    v = study.v
    data = v.FoldData(v.ETH3DAll(), 0)
    checkpoint = torch.load(study.HERE / "runs/fold0_seed0.pt", weights_only=True)
    cal = v.CalibratedMotor()
    cal.load_state_dict(checkpoint["calibrator"])
    head = study.Head("contextual")
    head.load_state_dict(checkpoint["contextual"])
    original = v.ContextualGA(cal.calibrator, 32)
    original.encoder.load_state_dict(head.encoder.state_dict())
    original.gate.load_state_dict(head.head.state_dict())
    leaves, raw, calibrated, _ = study.cache(data, cal, "test")
    maximum = 0.
    for start in range(0, len(leaves), 1024):
        end = start + 1024
        observed = head(leaves[start:end], raw[start:end], calibrated[start:end])
        expected = original(leaves[start:end])
        maximum = max(maximum, float((observed-expected).abs().max()))
    assert maximum <= 1e-6, maximum
    result = {"scope": "fold 0 seed 0, all recorded held-out L32 windows; same parameters",
              "n_windows": len(leaves), "maximum_coordinate_difference": maximum,
              "checkpoint_sha256": study.sha(study.HERE / "runs/fold0_seed0.pt")}
    study.save(study.HERE / "cached_inference_check.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
