# Read-only Oxford matrix audit replay

Oxford results are excluded, not endorsed. Do not run the frozen training or
confirmation campaign from this export. The native admission verifier fails
closed because an author-identifying acquisition helper is intentionally omitted.

First run `python verify_claims.py`. This checks all available sealed files,
the omission ledger, scientific result arithmetic and checkpoint integrity.

For source-backed matrix replay, obtain the official VO and RTK data under
Oxford's terms. Place only the 14 development traversals listed in
`data/raw/oxford_robotcar/manifest.json` at its relative `vo` and `rtk` paths,
and the official SDK `extrinsics/ins.txt` at `data/raw/oxford_robotcar/extrinsics/ins.txt`.
Do not provide or inspect confirmation data. The audit independently hashes
every development CSV and reconstructs the same recorded-row windows.

Run `python replay_oxford_matrix_audit.py` (or pass `--data-root PATH` to an
existing official Oxford CSV root without copying recordings). This wrapper confines its source
verification override to the read-only audit; it never invokes the joint
workflow's training, release or confirmation functions. It checks all available
sealed files and explicitly accounts for the omitted acquisition helper. Its
temporary report must equal the archived audit result exactly; the original
report is not overwritten. No input interpolation, fitted frame correction,
changed model or new observation is used.

The unmodified `audit.py` is also retained with its original source hash. The
wrapper is necessary only for this anonymous export, and must not be reused
to bypass outcome admission in any future scientific study.
