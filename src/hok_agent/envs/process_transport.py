"""Spawned-process transport for the deterministic M0 mock service.

The learner process only owns a pipe endpoint.  The mock environment and RPC
dispatcher are constructed in the child process, so no mutable environment
state is shared with the learner.
"""

from __future__ import annotations

from multiprocessing import get_context
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess

from hok_agent.contracts import EnvironmentKind
from hok_agent.envs.mock import MockEnvironment
from hok_agent.envs.rpc import LocalRpcServer, TransportError


def _serve_mock(connection: Connection, max_steps_per_episode: int) -> None:
    """Own the service-side state until the parent sends the shutdown sentinel."""

    server = LocalRpcServer(
        MockEnvironment(max_steps_per_episode=max_steps_per_episode),
        expected_kind=EnvironmentKind.MOCK,
    )
    try:
        while True:
            request = connection.recv()
            if request is None:
                return
            if not isinstance(request, str):
                connection.send(server.handle("{}"))
                continue
            connection.send(server.handle(request))
    except EOFError:
        return
    finally:
        connection.close()


class ProcessJsonTransport:
    """A bounded request/reply transport to one spawned mock service process."""

    def __init__(self, connection: Connection, process: BaseProcess, *, timeout_seconds: float) -> None:
        self._connection = connection
        self._process = process
        self._timeout_seconds = timeout_seconds
        self._closed = False

    @classmethod
    def start_mock(cls, *, max_steps_per_episode: int, timeout_seconds: float = 5.0) -> ProcessJsonTransport:
        if max_steps_per_episode < 2:
            raise ValueError("max_steps_per_episode must be at least two")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        context = get_context("spawn")
        parent_connection, child_connection = context.Pipe(duplex=True)
        process = context.Process(
            target=_serve_mock,
            args=(child_connection, max_steps_per_episode),
            daemon=True,
            name="hok-agent-v5-mock-service",
        )
        process.start()
        child_connection.close()
        return cls(parent_connection, process, timeout_seconds=timeout_seconds)

    @property
    def service_pid(self) -> int | None:
        return self._process.pid

    @property
    def is_alive(self) -> bool:
        return not self._closed and self._process.is_alive()

    def request(self, payload: str) -> str:
        if self._closed:
            raise TransportError("TRANSPORT_CLOSED", "mock service transport is closed")
        if not self._process.is_alive():
            raise TransportError("SERVICE_PROCESS_EXITED", "mock service process is not alive")
        try:
            self._connection.send(payload)
            if not self._connection.poll(self._timeout_seconds):
                raise TransportError("SERVICE_TIMEOUT", "mock service did not respond before timeout")
            response = self._connection.recv()
        except (BrokenPipeError, EOFError, OSError) as error:
            raise TransportError("TRANSPORT_FAILURE", "mock service transport failed") from error
        if not isinstance(response, str):
            raise TransportError("INVALID_TRANSPORT_RESPONSE", "mock service response is not text")
        return response

    def close(self) -> None:
        """Stop only this owned child process and release its pipe endpoint."""

        if self._closed:
            return
        try:
            if self._process.is_alive():
                self._connection.send(None)
                self._process.join(self._timeout_seconds)
                if self._process.is_alive():
                    self._process.terminate()
                    self._process.join(self._timeout_seconds)
        except (BrokenPipeError, EOFError, OSError):
            pass
        finally:
            self._connection.close()
            self._closed = True
