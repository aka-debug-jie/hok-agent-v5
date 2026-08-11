from __future__ import annotations

from pathlib import Path

from hok_agent.evaluation.smoke import run_mock_benchmark


def test_mock_benchmark_reports_positive_throughput(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config = config_dir / "run_smoke_v1.yaml"
    config.write_text(
        """environment:
  seed_start: 10
  episodes: 2
  max_steps_per_episode: 8
artifacts:
  output_root: artifacts/runs
""",
        encoding="utf-8",
    )
    result = run_mock_benchmark(config, episodes=2)
    assert result["environment_kind"] == "mock"
    assert float(result["env_steps_per_second"]) > 0.0
