"""Construction helper for the M0 local JSON RPC service stub.

This file deliberately exposes only the deterministic mock. Replacing it with an
authorized adapter is a later M1 task after a successful external preflight.
"""

from hok_agent.envs import LocalRpcServer, MockEnvironment


def build_m0_mock_server() -> LocalRpcServer:
    return LocalRpcServer(MockEnvironment())
