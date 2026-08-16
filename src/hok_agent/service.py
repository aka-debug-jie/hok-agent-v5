from __future__ import annotations

import multiprocessing as mp
from multiprocessing.connection import Connection
from pathlib import Path
from typing import cast

from hok_agent.arena import DEFAULT_CONFIG, FactorizedAction, PixelArena


class ServiceError(RuntimeError): pass  # noqa: E701


def _server(connection: Connection, config_path: str) -> None:
    arena = PixelArena(Path(config_path))
    while True:
        request = cast(dict[str, object], connection.recv())
        operation = str(request["operation"])
        try:
            if operation == "health":
                result = arena.health(); result["process_id"] = mp.current_process().pid  # noqa: E702
            elif operation == "reset":
                result = arena.reset(int(cast(int, request["seed"])))
            elif operation == "step":
                result = arena.step(
                    FactorizedAction.from_dict(cast(dict[str, object], request["blue_action"])),
                    FactorizedAction.from_dict(cast(dict[str, object], request["red_action"])))
            elif operation == "close":
                connection.send({"ok": True, "result": {"closed": True}}); break  # noqa: E702
            else:
                raise ValueError("unknown operation")
            connection.send({"ok": True, "result": result})
        except (KeyError, TypeError, ValueError) as exc:
            connection.send({"ok": False, "error": str(exc)})
    connection.close()


class PixelArenaService:
    def __init__(self, config_path: Path = DEFAULT_CONFIG) -> None:
        context = mp.get_context("spawn"); parent, child = context.Pipe()  # noqa: E702
        self._connection = parent
        self._process = context.Process(target=_server, args=(child, str(config_path)), daemon=True)
        self._process.start(); child.close(); self._closed = False  # noqa: E702

    def _request(self, operation: str, **payload: object) -> dict[str, object]:
        if self._closed: raise ServiceError("service is closed")  # noqa: E701
        self._connection.send({"operation": operation, **payload})
        response = cast(dict[str, object], self._connection.recv())
        if not bool(response["ok"]): raise ServiceError(str(response["error"]))  # noqa: E701
        return cast(dict[str, object], response["result"])

    def health(self) -> dict[str, object]: return self._request("health")  # noqa: E704

    def reset(self, seed: int) -> dict[str, object]:
        return self._request("reset", seed=seed)

    def step(self, blue_action: FactorizedAction,
             red_action: FactorizedAction) -> dict[str, object]:
        return self._request("step", blue_action=blue_action.to_dict(),
                             red_action=red_action.to_dict())

    def close(self) -> None:
        if self._closed: return  # noqa: E701
        if self._process.is_alive(): self._request("close")  # noqa: E701
        self._closed = True; self._connection.close()  # noqa: E702
        self._process.join(timeout=2)
        if self._process.is_alive():
            self._process.terminate(); self._process.join(timeout=2)  # noqa: E702

    def __enter__(self) -> PixelArenaService: return self  # noqa: E704

    def __exit__(self, *_args: object) -> None: self.close()  # noqa: E704
