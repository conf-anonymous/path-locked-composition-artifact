"""Fail-closed shared Oxford execution boundary while the joint runner is pending.

There is intentionally no environment-variable, file, or command-line bypass.
Replace this pending implementation only with the tested, sealed joint release
workflow required by the unchanged Experiment 031 protocol.
"""


def require_confirmation_release():
    raise RuntimeError(
        "Oxford confirmation remains sealed: Experiment 031's joint release "
        "workflow is not implemented and verified. Neither study may open confirmation."
    )


def require_development_ready():
    raise RuntimeError(
        "Oxford outcome inspection is not enabled: implement, test on public "
        "recorded ETH3D inputs, and seal the Experiment 031 runner first. "
        "Schema validation and manifest preparation are permitted."
    )
