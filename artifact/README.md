# Scientific evidence directory

This directory preserves the composition paper's scientific implementations,
protocols, checkpoints and retained outcomes. Start with the repository-level
[README](../README.md), [reproduction guide](../docs/REPRODUCING.md),
[environment guide](../docs/DATA_AND_ENVIRONMENTS.md) and
[claim map](../docs/CLAIMS.md).

`python verify_claims.py` runs the original stored-evidence checks from this
directory after installing the audit dependencies. The repository-level
`python reproduce.py audit` additionally checks complete membership and bytes
before and after execution and writes a separate execution log.

`README_PRE_CONSOLIDATION.md` is retained as historical protocol context and
detailed command documentation. Its earlier execution and manuscript status
statements are not the current state. KITTI has been evaluated on study drives
09-10; Oxford confirmation has not been opened. The top-level `PROTOCOL.md`
is the original MovieLens protocol, not a protocol governing every dataset.

The manuscript and publication files are submitted separately and are not
included here. `CONTENTS_SHA256.json` binds all current files in this directory
except itself. Historical source/input seals remain unchanged.
