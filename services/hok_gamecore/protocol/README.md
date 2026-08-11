# Protocol boundary

The initial protocol is JSON envelopes with `protocol_version: 1` and four methods:
`health`, `reset`, `step`, and `close`. The local implementation is deliberately
in-process so that M0 can test serialization and service isolation without an
external GameCore binary. This placeholder belongs only to the inactive optional
calibration track; replacing the transport requires a new authorized task and must
not change contract payloads.
