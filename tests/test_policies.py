# ruff: noqa: E501, E702
from hok_agent.arena import move_action, wait_action
from hok_agent.policies import NullPolicy, RandomPolicy, ScriptedPolicy


def test_baselines_are_deterministic_test_drivers() -> None:
    legal = (wait_action(), move_action("forward"), move_action("backward"))
    assert NullPolicy().select("blue", legal) == wait_action()
    assert ScriptedPolicy().select("blue", legal) == move_action("forward")
    first = RandomPolicy(9, "blue"); second = RandomPolicy(9, "blue")
    assert [first.select("blue", legal) for _ in range(20)] == [second.select("blue", legal) for _ in range(20)]
