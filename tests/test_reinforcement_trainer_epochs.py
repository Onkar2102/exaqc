"""End-to-end ``ReinforcementLearningTrainer.train`` tests over one+ episodes.

These tests drive the full training loop (a fixed number of training
episodes) of each RL algorithm on the deterministic test environment and
check the trainer's bookkeeping:

* per-episode metrics are recorded in ``genome.metadata``;
* ``best_training_metrics`` / ``best_validation_metrics`` are populated with
  finite returns; and
* the run completes for every target / trainer / encoder / decoder
  combination.

They complement ``test_reinforcement_trainer_gradients.py`` (which verifies
gradient flow through the three genome stages) by covering the outer loop,
the evaluation/best-snapshot bookkeeping, and the encoder/decoder variety.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from src.circuits.circuit import CircuitGenome

# from src.examples.reinforcement_learning import make_environment

from tests.reinforcement_trainer_test_utils import (
    ENCODER_DECODER_PAIRS,
    TRAINER_NAMES,
    build_rl_genome,
    build_trainer,
    make_test_environment,
)

TARGETS: tuple[str, ...] = ("pennylane", "qiskit")


def _assert_return_metrics(metrics: dict[str, float]) -> None:
    """Asserts a return-metrics dict has finite ``return_mean`` and best return.

    Args:
        metrics: A ``best_training_metrics`` or ``best_validation_metrics``
            dict recorded by the trainer.
    """

    assert "return_mean" in metrics
    assert math.isfinite(metrics["return_mean"])
    assert "best_episode_return" in metrics
    assert math.isfinite(metrics["best_episode_return"])


@pytest.mark.parametrize("trainer_name", TRAINER_NAMES)
@pytest.mark.parametrize("target", TARGETS)
def test_train_records_per_episode_and_best_metrics(
    target: str, trainer_name: str
) -> None:
    """Training records per-episode metrics and finite best-metric summaries.

    Args:
        target: Either ``"pennylane"`` or ``"qiskit"``.
        trainer_name: The RL algorithm to exercise.
    """

    trainer = build_trainer(trainer_name)
    genome, observation_features = build_rl_genome(
        genome_number=1,
        target=target,
        complexity="shallow",
        encoder_name="linear",
        decoder_name="linear",
        trainer=trainer,
    )
    environment = make_test_environment(observation_features)

    trainer.train(genome, environment)

    episode_metrics = genome.metadata["training_episode_metrics"]
    assert len(episode_metrics) == genome.hyperparameters["episodes"]
    for entry in episode_metrics:
        assert "episode" in entry
        assert "return" in entry
        assert math.isfinite(entry["return"])

    _assert_return_metrics(genome.metadata["best_training_metrics"])
    _assert_return_metrics(genome.metadata["best_validation_metrics"])
    # the deterministic env yields a constant +1 per step, so returns are >= 0
    assert genome.metadata["best_validation_metrics"]["return_mean"] >= 0.0


@pytest.mark.parametrize("encoder_name,decoder_name", ENCODER_DECODER_PAIRS)
@pytest.mark.parametrize("target", TARGETS)
def test_train_runs_across_encoder_decoder_combinations(
    target: str, encoder_name: str, decoder_name: str
) -> None:
    """Training completes for both trainable and stateless coder pairs.

    Uses REINFORCE (which needs no value output) so the ``identity``/
    ``clipped`` pair -- where the decoder has no trainable parameters -- is a
    valid configuration.

    Args:
        target: Either ``"pennylane"`` or ``"qiskit"``.
        encoder_name: Either ``"identity"`` or ``"linear"``.
        decoder_name: Either ``"clipped"`` or ``"linear"``.
    """

    trainer = build_trainer("reinforce")
    genome, observation_features = build_rl_genome(
        genome_number=2,
        target=target,
        complexity="shallow",
        encoder_name=encoder_name,
        decoder_name=decoder_name,
        trainer=trainer,
    )
    environment = make_test_environment(observation_features)

    trainer.train(genome, environment)

    assert (
        len(genome.metadata["training_episode_metrics"])
        == genome.hyperparameters["episodes"]
    )
    _assert_return_metrics(genome.metadata["best_validation_metrics"])


@pytest.mark.parametrize("target", TARGETS)
def test_train_with_no_trainable_parameters_only_evaluates(target: str) -> None:
    """A parameter-free genome is evaluated rather than trained.

    With an ``IdentityEncoder``, a ``ClippedDecoder``, and no parametric
    gates, the genome's hybrid model has zero trainable parameters, so the
    trainer should take its evaluation-only path: no per-episode training
    metrics, but best-metric summaries still recorded.

    Args:
        target: Either ``"pennylane"`` or ``"qiskit"``.
    """

    trainer = build_trainer("reinforce")
    genome, observation_features = build_rl_genome(
        genome_number=3,
        target=target,
        complexity="shallow",
        encoder_name="identity",
        decoder_name="clipped",
        trainer=trainer,
        include_parametric=False,
    )
    environment = make_test_environment(observation_features)

    trainer.train(genome, environment)

    assert genome.metadata["training_episode_metrics"] == []
    _assert_return_metrics(genome.metadata["best_training_metrics"])
    _assert_return_metrics(genome.metadata["best_validation_metrics"])


@pytest.mark.parametrize("trainer_name", TRAINER_NAMES)
def test_train_respects_episode_count_from_hyperparameters(trainer_name: str) -> None:
    """The number of recorded training episodes matches the hyperparameter.

    Args:
        trainer_name: The RL algorithm to exercise.
    """

    trainer = build_trainer(trainer_name)
    genome, observation_features = build_rl_genome(
        genome_number=4,
        target="pennylane",
        complexity="minimal",
        encoder_name="linear",
        decoder_name="linear",
        trainer=trainer,
    )
    genome.hyperparameters["episodes"] = 3
    environment = make_test_environment(observation_features)

    trainer.train(genome, environment)

    assert isinstance(genome, CircuitGenome)
    assert len(genome.metadata["training_episode_metrics"]) == 3


def test_frozenlake_is_flagged_deterministic_only_when_not_slippery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FrozenLake is deterministic unless slippery; other envs are stochastic."""

    import sys
    import types

    mock_master_worker = types.ModuleType("src.evolution.master_worker")
    mock_master_worker.master_worker = lambda *args, **kwargs: None

    monkeypatch.setitem(
        sys.modules,
        "src.evolution.master_worker",
        mock_master_worker,
    )

    from src.examples.reinforcement_learning import make_environment

    assert make_environment("frozenlake").deterministic is True
    assert make_environment("frozenlake", is_slippery=True).deterministic is False
    assert make_environment("cartpole").deterministic is False


@pytest.mark.parametrize("deterministic", [False, True])
def test_evaluate_runs_single_episode_for_deterministic_environment(
    deterministic: bool,
) -> None:
    """``evaluate`` runs one episode for a deterministic env, else eval_episodes.

    Greedy evaluation of a deterministic environment yields identical episodes,
    so only one is run; a stochastic environment runs the full
    ``eval_episodes`` count.

    Args:
        deterministic: Whether the environment is marked deterministic.
    """

    trainer = build_trainer("reinforce")
    genome, observation_features = build_rl_genome(
        genome_number=1,
        target="pennylane",
        complexity="minimal",
        encoder_name="linear",
        decoder_name="linear",
        trainer=trainer,
    )
    genome.initialize_model()
    hp = trainer.resolve_hyperparameters(genome)
    assert hp.eval_episodes > 1  # so the two cases differ

    environment = dataclasses.replace(
        make_test_environment(observation_features), deterministic=deterministic
    )

    # count how many episodes evaluate() actually rolls (one env.make per episode)
    episode_count = 0
    original_make = environment.make

    def counting_make(*args, **kwargs):
        nonlocal episode_count
        episode_count += 1
        return original_make(*args, **kwargs)

    environment.make = counting_make
    trainer.evaluate(genome, environment, hp)

    assert episode_count == (1 if deterministic else hp.eval_episodes)
