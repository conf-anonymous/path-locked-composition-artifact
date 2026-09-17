"""Exact product-cache update audit using calibrated real-recording operands."""
import json
import time

import torch

import run as study


@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    v = study.v
    data = v.FoldData(v.ETH3DAll(), 0)
    path = study.HERE / "runs/fold0_seed0.pt"
    saved = torch.load(path, weights_only=True)
    calibrator = v.CalibratedMotor().double()
    calibrator.load_state_dict(saved["calibrator"])
    leaves, _, _ = data.tensors[("test", 32)]
    corrected = calibrator.calibrator(leaves)
    tree = [None] * 64
    tree[32:] = list(leaves.unbind(1))
    for i in range(31, 0, -1):
        tree[i] = v.base.motor_product(tree[2*i], tree[2*i+1])
    evolving = list(leaves.unbind(1))
    max_t, max_r, max_coordinate = 0., 0., 0.
    update_calls = 0
    # Recalibrating predicted states is a model update, not an alteration to
    # recorded sensor observations or reference labels. No synthetic data.
    for position in range(32):
        evolving[position] = corrected[:, position]
        tree[32+position] = corrected[:, position]
        index = (32+position) // 2
        while index:
            tree[index] = v.base.motor_product(tree[2*index], tree[2*index+1])
            update_calls += 1
            index //= 2
        full = v.base.reduce_states(evolving, v.base.motor_product, "balanced")
        max_coordinate = max(max_coordinate, float((tree[1]-full).abs().max()))
        q0, t0 = v.base.decode_motor(tree[1])
        q1, t1 = v.base.decode_motor(full)
        relative = v.base.qmul(v.base.qconj(q0), q1)
        angle = 2*torch.atan2(relative[:, 1:].norm(dim=-1), relative[:, 0].abs())
        max_t = max(max_t, float((t0-t1).norm(dim=-1).max()))
        max_r = max(max_r, float(angle.max()))
    result = {"scope": "exact product states only, not GRU context; fold 0 seed 0",
        "checkpoint_sha256": study.sha(path), "n_recorded_windows": len(leaves),
        "updates_per_window": 32, "initial_product_calls": 31,
        "cached_product_calls_all_updates": update_calls,
        "full_recompute_product_calls_all_updates": 32*31,
        "max_coordinate_difference": max_coordinate,
        "max_translation_difference_m": max_t, "max_rotation_difference_rad": max_r,
        "operation_count_note": "batched product calls, same count per window; not measured wall-clock speedup"}
    assert update_calls == 160 and max_t < 1e-10 and max_r < 1e-10
    study.save(study.HERE / "cache_deployment.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
