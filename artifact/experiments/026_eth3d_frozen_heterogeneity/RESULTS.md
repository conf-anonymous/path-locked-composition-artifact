# Post-hoc ETH3D heterogeneity result

This diagnostic reads only immutable Experiment 023--024 JSON. It does not
train, select, or alter a model or endpoint.

Contextual GA beats the fully calibrated direct GRU on translation in 11/16
families. Its family-median difference is -0.0669 m, close to the family mean
of -0.0632 m. An exact enumeration of all 65,536 family-level sign assignments
gives one-sided `p=0.0174` and two-sided `p=0.0348`. This test was added post hoc
and complements rather than replaces the prespecified family-bootstrap interval.

The mean contextual advantage remains negative after removing every family in
turn; the least favorable leave-one-family-out mean is -0.0468 m when
`vicon_light` is removed. It also remains negative after removing the one, two,
or three families with the largest raw errors. Removing the three largest
(`camera_shake`, `mannequin`, and `cables`) yields -0.0926 m versus the matched
direct control and -0.8836 m versus raw composition. The primary conclusion is
therefore not created by the largest raw-odometry failures.

The mechanisms differ by regime. Raw error strongly predicts improvement over
raw composition (Spearman 0.935), but it does not predict improvement over the
matched direct control (Spearman -0.691): on the two highest-drift families,
`camera_shake` and `mannequin`, both learned methods rescue raw odometry and the
direct model is better. Contextual GA's comparative translation advantage is
strongest on several lower-drift families, including `vicon_light`, `planar`,
`table`, `sofa`, and `repetitive`.

Contextual GA has lower rotation error in 10/16 families, but its family-equal
mean is 1.565 degrees worse because `camera_shake` has a 65.7-degree contextual
disadvantage. This confirms that the paper must limit the matched-control claim
to translation and treat rotation as a documented failure mode.
