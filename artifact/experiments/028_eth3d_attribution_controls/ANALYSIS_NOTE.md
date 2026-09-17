# Numerical analysis check

Before manuscript integration, the exhaustive fold sign-flip calculation was
found to use an absolute 1e-15 equality tolerance, too small for the degree-scale
rotation contrast after different summation orders. The derived analysis now
uses a scale-aware 1e-12 tolerance and verifies that an exhaustive two-sided
five-block enumeration includes at least the observed assignment and its
negative (minimum p=2/32). No training source, fitted model, per-sequence error,
selection decision or original experiment result was changed.
