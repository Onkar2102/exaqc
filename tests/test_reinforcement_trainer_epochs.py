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

import math

import pytest

from src.circuits.circuit import CircuitGenome

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
