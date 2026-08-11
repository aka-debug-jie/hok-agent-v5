# hok-gamecore-service

This directory is a transport-isolated placeholder for a future **authorized**
GameCore adapter. It contains no Tencent binary, license, account, or client-control
code. The Python 3.11 learner talks only to the versioned RPC contract in
`src/hok_agent/envs/rpc.py`.

Until an authorized GameCore path and license are supplied out of band, use only the
deterministic mock service. A future upstream SDK service must use its own Python
environment (currently expected to be `<3.10`) and must fail closed on identity,
license, schema, or tick mismatches.
