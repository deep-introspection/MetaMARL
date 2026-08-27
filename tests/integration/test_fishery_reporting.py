"""End-to-end smoke test of the fishery run with the CSV reporter (logging branch)."""

import csv

import numpy as np
import pytest
import ray

from examples.bilevel_fishery.debug import build_config, parse_args


@pytest.mark.integration
def test_bilevel_smoke_writes_csv_reports(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    ray.shutdown()
    cfg = build_config(
        parse_args(
            [
                "--outer-iters",
                "2",
                "--train-iters",
                "2",
                "--num-agents",
                "2",
                "--horizon",
                "20",
                "--num-candidates",
                "2",
                "--num-eval-seeds",
                "1",
                "--num-cpus",
                "2",
                "--reporter",
                "csv",
                "--output-dir",
                str(tmp_path),
                "--project",
                "smoke",
            ]
        )
    )
    try:
        result = cfg.build_optimizer().run()
    finally:
        ray.shutdown()

    assert result["outer_iters"] == 2 and np.isfinite(result["best_fitness"])

    files = {p.name: p for p in (tmp_path / "smoke").rglob("*.csv")}
    assert "Fitness_over_generations.csv" in files
    assert "Candidate_fitness_all_candidates_.csv" in files
    assert "Train_reward.csv" in files
    assert "Fish_biomass.csv" in files
    assert "Mean_reward_across_agents_1_std.csv" in files

    with files["Fitness_over_generations.csv"].open() as f:
        rows = list(csv.DictReader(f))
    assert {r["series"] for r in rows} == {
        "fitness_mean",
        "fitness_best",
        "best_fitness_global",
    }
    assert sorted(
        int(r["x"])
        for r in rows
        if r["series"] == "sigma" or r["series"] == "fitness_mean"
    ) == [1, 2]

    with files["Mean_reward_across_agents_1_std.csv"].open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 20 and all(r["error"] != "" for r in rows)
