# Frozen-output ETH3D heterogeneity analysis

Defined on 2026-09-05 after the ETH3D model freeze. This analysis reads only
immutable Experiment 023 and 024 result records. It performs no training,
checkpoint selection, endpoint selection, data transformation, or generated
observation construction.

At the already-primary length 32, report every one of the 16 capture families,
translation differences against raw composition and the fully matched direct
GRU, rotation differences against the direct GRU, Spearman associations with
raw translation error, an exact family sign-flip test, all leave-one-family-out
means, and means after removing the one, two, and three highest-raw-error
families. These are explicitly post hoc diagnostics and cannot replace the
prespecified family bootstrap.
