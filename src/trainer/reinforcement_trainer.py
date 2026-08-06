"""Modular reinforcement-learning trainers for evolved quantum genomes.

This module mirrors :mod:`src.trainer.supervised_trainer` but for
reinforcement learning. Where ``SupervisedTrainer`` consumes dataloaders,
loss functions and metrics and calls ``genome.forward`` on labelled samples,
the trainers here drive a ``CircuitGenome`` as a *policy* (or *value*)
network inside a Gymnasium environment.

Every trainer uses the same modular ``CircuitGenome`` interface the
supervised path uses:

* ``genome.initialize_model()`` builds the ``hybrid_model`` (encoder ->
  quantum layer -> decoder);
* ``genome.forward(observation)`` produces one output value per action
  (interpreted as policy logits or Q-values);
* ``genome.hybrid_model.parameters()`` are the only trainable parameters;
* ``genome.set_parameters(state_dict)`` restores the best-performing weights.

Terminology
    The trainers use these terms consistently:

    * **step**: one interaction with the environment (observe -> act ->
      receive a reward).
    * **episode**: one full rollout through the environment, from ``reset``
      until a terminal state or ``max_steps`` steps -- i.e. a sequence of
      steps.
    * **epoch**: one weight update (a single ``optimizer.step()``).

    The outer training loop runs ``episodes`` episodes. How many epochs
    (weight updates) occur per episode depends on the algorithm:

    * REINFORCE / actor-critic: run one episode, then perform one epoch;
    * Q-learning / SARSA: run one episode, performing one epoch at every
      environment step;
    * PPO: collect several episodes into a rollout, then perform many epochs
      across ``ppo_passes`` passes over that rollout. PPO is the one algorithm
      whose outer-loop iteration spans more than one episode (see
      :class:`~src.trainer.ppo_trainer.PPOTrainer`).

Value-based advantage methods (actor-critic, PPO) need a scalar state value
in addition to the per-action policy outputs. Rather than owning a separate
value head (whose weights would live outside the genome -- never serialized,
never recombined by the encoder/decoder crossover operators, and re-created
from scratch every evaluation), those trainers ask the genome's decoder for
one *extra* output. The decoder is sized to ``n_actions + n_value_outputs``,
so the value estimate is just another row of the decoder's linear layer and
is therefore part of ``genome.hybrid_model`` -- evolved by
``crossover_encoder_decoder`` / ``torch_simplex_crossover`` and preserved
across ``to_dict``/``from_dict`` exactly like the policy weights. This does
assume a decoder whose outputs are unconstrained linear features (the default
``LinearDecoder``); a normalizing decoder such as ``ClippedDecoder`` is not
appropriate for these algorithms.

The classical observation encoder (raw env observation -> fixed-length
feature vector) is kept separate from the genome's learnable ``Encoder`` so
existing ``LinearEncoder``/``IdentityEncoder`` classes can be reused
unchanged: the observation encoder maps a Gym observation into the feature
vector the genome's ``Encoder`` then embeds into the quantum circuit.

This module provides the shared infrastructure -- the environment
abstraction, observation encoders, resolved hyperparameters, RL math helpers,
and the abstract base :class:`ReinforcementLearningTrainer`. The concrete
algorithms each live in their own module and subclass the base here:

* :class:`~src.trainer.reinforce_trainer.ReinforceTrainer` -- Monte-Carlo
  policy gradient (REINFORCE).
* :class:`~src.trainer.actor_critic_trainer.ActorCriticTrainer` -- on-policy
  advantage actor-critic.
* :class:`~src.trainer.ppo_trainer.PPOTrainer` -- proximal policy
  optimization with GAE.
* :class:`~src.trainer.q_learning_trainer.QLearningTrainer` -- semi-gradient
  Q-learning / SARSA.

They are collected by name in
:data:`src.trainer.rl_trainer_registry.TRAINER_REGISTRY`. All four target
discrete action spaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional

import math
import numpy as np
import torch

from loguru import logger
from torch import Tensor

import gymnasium as gym

from src.circuits.circuit import CircuitGenome

# ---------------------------------------------------------------------------
# Environment abstraction
# ---------------------------------------------------------------------------


@dataclass
class RLEnvironment:
    """A pluggable description of a reinforcement-learning environment.

    This wraps a Gymnasium environment id together with everything a trainer
    needs to run it against a genome policy: the number of discrete actions,
    the size of the encoded observation vector, and a callable that turns a
    raw environment observation into that fixed-length feature tensor.

    Keeping the observation encoder here (rather than inside the genome)
    means the genome's own learnable ``Encoder`` only ever sees a clean,
    fixed-length feature vector, so the existing ``LinearEncoder`` /
    ``IdentityEncoder`` classes work without modification.

    Attributes:
        env_id: Gymnasium environment id (e.g. ``"CartPole-v1"``).
        n_actions: Number of discrete actions.
        n_observation_features: Length of the encoded observation vector,
            i.e. the number of inputs the genome's ``Encoder`` expects.
        obs_encoder: Callable mapping a raw observation into a float tensor
            of shape ``(n_observation_features,)``.
        env_kwargs: Optional keyword arguments passed to ``gym.make``.
        deterministic: Whether the environment is fully deterministic (fixed
            initial state and transitions). When True, a greedy policy
            produces the same episode every time regardless of seed, so
            greedy evaluation runs a single episode instead of
            ``eval_episodes`` identical ones (see
            :meth:`ReinforcementLearningTrainer.evaluate`).
    """

    env_id: str
    n_actions: int
    n_observation_features: int
    obs_encoder: Callable[[Any], Tensor]
    env_kwargs: Optional[dict[str, Any]] = None
    deterministic: bool = False

    def make(self) -> gym.Env:
        """Instantiates the underlying Gymnasium environment.

        Returns:
            A new ``gym.Env`` instance.
        """

        return gym.make(self.env_id, **(self.env_kwargs or {}))

    def encode(self, observation: Any) -> Tensor:
        """Encodes a raw observation into the genome's input feature vector.

        Args:
            observation: A raw observation returned by the environment.

        Returns:
            A float tensor of shape ``(n_observation_features,)``.
        """

        return self.obs_encoder(observation)


def box_observation_encoder(
    scales: Optional[np.ndarray] = None,
) -> Callable[[Any], Tensor]:
    """Builds an encoder for continuous ``Box`` observations.

    Args:
        scales: Optional per-dimension scale factors. When provided, each
            observation dimension is divided by its scale and clipped to
            ``[-1, 1]``; otherwise the raw observation is returned as a float
            tensor (the genome's learnable encoder can then rescale it).

    Returns:
        A callable mapping an observation into a float tensor.
    """

    scale_array = None if scales is None else np.asarray(scales, dtype=np.float32)

    def encode(observation: Any) -> Tensor:
        values = np.asarray(observation, dtype=np.float32).reshape(-1)
        if scale_array is not None:
            values = np.clip(values / scale_array, -1.0, 1.0)
        return torch.tensor(values, dtype=torch.float32)

    return encode


def onehot_observation_encoder(n_states: int) -> Callable[[Any], Tensor]:
    """Builds a one-hot encoder for discrete integer observations.

    Useful for tabular environments such as ``FrozenLake`` whose observation
    is a single integer state index.

    Args:
        n_states: Total number of discrete states.

    Returns:
        A callable mapping an integer state into a one-hot float tensor of
        shape ``(n_states,)``.
    """

    def encode(observation: Any) -> Tensor:
        vector = torch.zeros(n_states, dtype=torch.float32)
        vector[int(observation)] = 1.0
        return vector

    return encode


# ---------------------------------------------------------------------------
# Resolved hyperparameters
# ---------------------------------------------------------------------------


@dataclass
class RLHyperparameters:
    """Per-genome training hyperparameters resolved for a single run.

    A trainer reads these from ``genome.hyperparameters`` (so the
    evolutionary search can mutate them per genome), falling back to the
    trainer's own defaults when a key is absent.

    Attributes:
        episodes: Number of training episodes (outer-loop iterations).
        learning_rate: Adam learning rate.
        gamma: Reward discount factor.
        max_steps: Maximum number of steps per episode.
        eval_episodes: Number of episodes used for greedy evaluation.
        seed: Base random seed.
        log_every: Logging / evaluation frequency, in episodes.
        baseline: REINFORCE advantage baseline (``"mean"`` or ``"none"``).
        entropy_coef: Entropy-bonus coefficient.
        value_coef: Weight on the value loss (actor-critic / PPO).
        gae_lambda: Generalized Advantage Estimation lambda (PPO).
        rollout_steps: Environment steps collected per PPO episode (a rollout
            spanning one or more environment episodes).
        ppo_passes: Number of passes over a collected PPO rollout; each pass
            performs several minibatch epochs (weight updates). PPO literature
            often calls these passes "epochs"; renamed here so "epoch" refers
            only to a single weight update.
        ppo_minibatch: PPO minibatch size (steps per weight update).
        ppo_clip: PPO clip range.
        epsilon: Initial epsilon for epsilon-greedy exploration (value-based).
        epsilon_min: Minimum epsilon.
        epsilon_decay: Per-episode multiplicative epsilon decay.
    """

    episodes: int = 60
    learning_rate: float = 1e-2
    gamma: float = 0.99
    max_steps: int = 500
    eval_episodes: int = 10
    seed: int = 0
    log_every: int = 10
    baseline: str = "mean"
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    gae_lambda: float = 0.95
    rollout_steps: int = 512
    ppo_passes: int = 4
    ppo_minibatch: int = 128
    ppo_clip: float = 0.2
    epsilon: float = 0.2
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995


# ---------------------------------------------------------------------------
# Small self-contained RL math helpers
# ---------------------------------------------------------------------------


def discounted_returns(rewards: list[float], gamma: float) -> Tensor:
    """Computes discounted Monte-Carlo returns for a reward sequence.

    Args:
        rewards: List of scalar step rewards, in time order.
        gamma: Discount factor.

    Returns:
        A float tensor of shape ``(len(rewards),)`` of discounted returns.
    """

    returns: list[float] = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.append(running)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


def gae_advantages(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    *,
    gamma: float,
    lam: float,
) -> tuple[Tensor, Tensor]:
    """Computes Generalized Advantage Estimation advantages and returns.

    Args:
        rewards: Reward tensor of shape ``(T,)``.
        values: Value estimates of shape ``(T,)``.
        dones: Done flags of shape ``(T,)`` with values in ``{0.0, 1.0}``.
        gamma: Discount factor.
        lam: GAE lambda.

    Returns:
        A tuple ``(advantages, returns)``, each of shape ``(T,)``, where
        ``returns = advantages + values``.
    """

    n_steps = rewards.numel()
    advantages = torch.zeros(n_steps, dtype=torch.float32)
    last_advantage = 0.0
    next_value = 0.0

    for t in reversed(range(n_steps)):
        mask = 1.0 - float(dones[t].item())
        delta = (
            float(rewards[t].item())
            + gamma * next_value * mask
            - float(values[t].item())
        )
        last_advantage = delta + gamma * lam * mask * last_advantage
        advantages[t] = last_advantage
        next_value = float(values[t].item())

    returns = advantages + values
    return advantages, returns


def _normalize(x: Tensor, eps: float = 1e-8) -> Tensor:
    """Normalizes a tensor to zero mean and unit variance.

    Args:
        x: Input tensor.
        eps: Numerical-stability constant.

    Returns:
        The normalized tensor (unchanged if it has fewer than two elements).
    """

    if x.numel() < 2:
        return x
    return (x - x.mean()) / (x.std() + eps)


# ---------------------------------------------------------------------------
# Base trainer
# ---------------------------------------------------------------------------


class ReinforcementLearningTrainer(ABC):
    """Base class for reinforcement-learning trainers over circuit genomes.

    Subclasses implement a single algorithm update in :meth:`run_update`, and
    declare via :attr:`n_value_outputs` how many extra decoder outputs the
    algorithm needs (e.g. a scalar state value). This base class owns the
    generic training scaffold that mirrors ``SupervisedTrainer.train``:

    * initialize the genome's hybrid model,
    * short-circuit to evaluation-only when there are no trainable
      parameters,
    * build the optimizer over the genome's parameters,
    * run the algorithm for ``episodes`` episodes, periodically evaluating and
      snapshotting the best-performing weights,
    * restore the best weights and record metrics into ``genome.metadata``.

    See the module docstring for how the "step", "episode", and "epoch"
    (one weight update) terms are used.

    Hyperparameters are read from ``genome.hyperparameters`` where present
    (so the search can mutate them), otherwise from the values passed to this
    constructor.

    Args:
        episodes: Number of training episodes (outer-loop iterations).
        learning_rate: Adam learning rate.
        gamma: Reward discount factor.
        max_steps: Maximum number of steps per episode.
        eval_episodes: Number of episodes used for greedy evaluation.
        seed: Base random seed.
        log_every: Logging / evaluation frequency, in episodes.
        entropy_coef: Entropy-bonus coefficient.
        baseline: Baseline used by REINFORCE (``"mean"`` or ``"none"``).
        value_coef: Weight on the value loss (actor-critic / PPO).
        gae_lambda: GAE lambda (PPO).
        rollout_steps: Environment steps collected per PPO episode.
        ppo_passes: Passes over a collected PPO rollout; each pass performs
            several minibatch epochs (weight updates).
        ppo_minibatch: PPO minibatch size (steps per weight update).
        ppo_clip: PPO clip range.
        epsilon: Initial epsilon for epsilon-greedy (value-based).
        epsilon_min: Minimum epsilon.
        epsilon_decay: Per-episode multiplicative epsilon decay.

    Class Attributes:
        n_value_outputs: How many extra decoder outputs (beyond
            ``n_actions``) this algorithm needs. ``0`` for policy-only /
            value-based methods; ``1`` for advantage methods that read a
            scalar state value out of the decoder. The example script sizes
            the genome's decoder as ``n_actions + n_value_outputs``.
    """

    #: Extra decoder outputs required beyond the per-action outputs (see above).
    n_value_outputs: int = 0

    def __init__(
        self,
        *,
        episodes: int = 60,
        learning_rate: float = 1e-2,
        gamma: float = 0.99,
        max_steps: int = 500,
        eval_episodes: int = 10,
        seed: int = 0,
        log_every: int = 10,
        entropy_coef: float = 0.0,
        baseline: str = "mean",
        value_coef: float = 0.5,
        gae_lambda: float = 0.95,
        rollout_steps: int = 512,
        ppo_passes: int = 4,
        ppo_minibatch: int = 128,
        ppo_clip: float = 0.2,
        epsilon: float = 0.2,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        self.defaults = RLHyperparameters(
            episodes=episodes,
            learning_rate=learning_rate,
            gamma=gamma,
            max_steps=max_steps,
            eval_episodes=eval_episodes,
            seed=seed,
            log_every=log_every,
            entropy_coef=entropy_coef,
            baseline=baseline,
            value_coef=value_coef,
            gae_lambda=gae_lambda,
            rollout_steps=rollout_steps,
            ppo_passes=ppo_passes,
            ppo_minibatch=ppo_minibatch,
            ppo_clip=ppo_clip,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
        )

    # -- hooks for subclasses -------------------------------------------------

    @abstractmethod
    def run_update(
        self,
        genome: CircuitGenome,
        environment: RLEnvironment,
        optimizer: torch.optim.Optimizer,
        episode_index: int,
        hp: RLHyperparameters,
    ) -> tuple[float, dict[str, float]]:
        """Runs one outer-loop training episode and its weight update(s).

        Called once per outer-loop episode by :meth:`train`. A subclass
        collects experience by rolling one or more episodes through the
        environment and performs one or more epochs (weight updates); see the
        module docstring for the per-algorithm breakdown.

        Args:
            genome: The genome being trained.
            environment: The environment being trained on.
            optimizer: The optimizer over the genome's parameters.
            episode_index: Zero-based index of this training episode, used to
                seed the environment reset.
            hp: Resolved hyperparameters.

        Returns:
            A tuple ``(episode_return, info)`` where ``episode_return`` is a
            representative episode return for logging/tracking (the mean over
            collected episodes for PPO) and ``info`` is a dict of extra scalar
            metrics recorded per episode.
        """

    # -- shared helpers -------------------------------------------------------

    def resolve_hyperparameters(self, genome: CircuitGenome) -> RLHyperparameters:
        """Resolves the hyperparameters to use for training a genome.

        Each field is taken from the genome's ``hyperparameters`` dict when
        present (so the evolutionary search can mutate it per genome), and
        otherwise from the defaults this trainer was constructed with. This is
        the same resolution :meth:`train` performs internally; it is public so
        callers can build the :class:`RLHyperparameters` needed to drive a
        single :meth:`run_update` (e.g. for a custom loop or a unit test)
        without reaching into private state.

        Args:
            genome: The genome whose ``hyperparameters`` dict is consulted.

        Returns:
            A fully-populated :class:`RLHyperparameters`.
        """

        source = getattr(genome, "hyperparameters", {}) or {}
        resolved = RLHyperparameters()
        for name in resolved.__dataclass_fields__:
            resolved.__dict__[name] = source.get(name, getattr(self.defaults, name))
        return resolved

    def policy_logits(
        self, genome: CircuitGenome, environment: RLEnvironment, observation: Any
    ) -> Tensor:
        """Computes policy logits (or Q-values) for a raw observation.

        Only the per-action outputs are returned; if the decoder also carries
        a trailing value output (see :attr:`n_value_outputs`) it is sliced
        off here.

        Args:
            genome: The genome policy.
            environment: The environment (provides observation encoding).
            observation: A raw environment observation.

        Returns:
            The genome's per-action output vector of shape ``(n_actions,)``.
        """

        output = genome.forward(environment.encode(observation))
        return output[: environment.n_actions]

    @staticmethod
    def split_policy_value(output: Tensor, n_actions: int) -> tuple[Tensor, Tensor]:
        """Splits a genome output into per-action logits and a scalar value.

        Used by advantage methods (actor-critic, PPO) whose decoder produces
        ``n_actions + 1`` outputs: the first ``n_actions`` are policy logits
        and the last is the state-value estimate.

        Args:
            output: The genome's raw output vector of shape
                ``(n_actions + 1,)``.
            n_actions: The number of discrete actions.

        Returns:
            A tuple ``(logits, value)`` where ``logits`` has shape
            ``(n_actions,)`` and ``value`` is a scalar tensor.
        """

        return output[:n_actions], output[n_actions]

    @torch.no_grad()
    def evaluate(
        self,
        genome: CircuitGenome,
        environment: RLEnvironment,
        hp: RLHyperparameters,
    ) -> dict[str, float]:
        """Evaluates the genome greedily over several episodes.

        Because evaluation is greedy (deterministic policy), the only source
        of variation between episodes is the environment. For a deterministic
        environment (``environment.deterministic``) every episode is therefore
        identical, so a single episode is run instead of ``eval_episodes``
        redundant copies.

        Args:
            genome: The genome policy to evaluate.
            environment: The environment to evaluate on.
            hp: Resolved hyperparameters.

        Returns:
            A dict with ``return_mean``, ``return_std`` and
            ``best_episode_return``.
        """

        n_episodes = 1 if environment.deterministic else hp.eval_episodes

        returns: list[float] = []
        for episode in range(n_episodes):
            env = environment.make()
            observation, _ = env.reset(seed=hp.seed + 10_000 + episode)
            episode_return = 0.0
            for _ in range(hp.max_steps):
                logits = self.policy_logits(genome, environment, observation)
                action = int(torch.argmax(logits).item())
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
                if terminated or truncated:
                    break
            env.close()
            returns.append(episode_return)

        return {
            "return_mean": float(np.mean(returns)) if returns else 0.0,
            "return_std": float(np.std(returns)) if returns else 0.0,
            "best_episode_return": float(np.max(returns)) if returns else 0.0,
        }

    def _clone_hybrid_state(self, genome: CircuitGenome) -> dict[str, Tensor]:
        """Snapshots the genome's hybrid-model weights.

        Args:
            genome: An initialized genome.

        Returns:
            A detached, cloned ``state_dict`` suitable for
            ``genome.set_parameters``.
        """

        with torch.no_grad():
            return {
                name: tensor.detach().clone()
                for name, tensor in genome.hybrid_model.state_dict().items()
            }

    # -- main entry point -----------------------------------------------------

    def train(self, genome: CircuitGenome, environment: RLEnvironment) -> None:
        """Trains a genome on an environment and records metrics.

        Runs ``hp.episodes`` training episodes (each delegating to
        :meth:`run_update`), evaluates periodically, and restores the
        best-evaluated weights. On completion the genome's ``metadata``
        contains ``training_episode_metrics`` (per-episode returns),
        ``best_training_metrics`` and ``best_validation_metrics``.

        Args:
            genome: The genome to train (its model is initialized here).
            environment: The environment to train on.
        """

        hp = self.resolve_hyperparameters(genome)

        genome.initialize_model()

        torch.manual_seed(hp.seed)
        np.random.seed(hp.seed)

        # All trainable parameters live in the genome's hybrid model (encoder,
        # quantum layer, decoder) -- including the value output for advantage
        # methods, which is an extra decoder row. There is no external head to
        # optimize, so everything trained here is also evolved by crossover
        # and preserved through genome serialization.
        trainable_parameters = list(genome.hybrid_model.parameters())

        genome.metadata["training_episode_metrics"] = []

        n_trainable = sum(p.numel() for p in trainable_parameters if p.requires_grad)
        if n_trainable == 0:
            # nothing to optimize -- just evaluate the untrained genome
            logger.info("genome has no trainable parameters; evaluating only.")
            evaluation = self.evaluate(genome, environment, hp)
            genome.metadata["best_training_metrics"] = {
                "return_mean": evaluation["return_mean"],
                "best_episode_return": evaluation["best_episode_return"],
            }
            genome.metadata["best_validation_metrics"] = evaluation
            return

        optimizer = torch.optim.Adam(
            trainable_parameters, lr=hp.learning_rate, weight_decay=0.0
        )

        recent_returns: list[float] = []
        best_return = -math.inf
        best_state = self._clone_hybrid_state(genome)
        best_evaluation = None
        eval_every = max(1, hp.log_every)
        best_episode = 0

        for episode in range(hp.episodes):
            episode_return, info = self.run_update(
                genome, environment, optimizer, episode, hp
            )
            recent_returns.append(episode_return)

            episode_metrics = {"episode": episode, "return": episode_return}
            episode_metrics.update(info)
            genome.metadata["training_episode_metrics"].append(episode_metrics)

            if (episode % eval_every == 0) or (episode == hp.episodes - 1):
                evaluation = self.evaluate(genome, environment, hp)
                logger.info(
                    f"[{type(self).__name__}] episode {episode:04d} "
                    f"train_return={episode_return:.1f} "
                    f"eval_return_mean={evaluation['return_mean']:.1f}"
                )
                if evaluation["return_mean"] > best_return:
                    best_return = evaluation["return_mean"]
                    best_evaluation = evaluation
                    best_state = self._clone_hybrid_state(genome)
                    best_episode = episode

        # restore the best-evaluated weights into the genome
        genome.set_parameters(best_state)

        if len(recent_returns) >= 20:
            train_tail = float(np.mean(recent_returns[-20:]))
        elif recent_returns:
            train_tail = float(np.mean(recent_returns))
        else:
            train_tail = 0.0

        genome.metadata["best_episode"] = best_episode
        genome.metadata["best_training_metrics"] = {
            "return_mean": train_tail,
            "best_episode_return": (
                float(np.max(recent_returns)) if recent_returns else 0.0
            ),
        }
        genome.metadata["best_validation_metrics"] = (
            best_evaluation
            if best_evaluation is not None
            else self.evaluate(genome, environment, hp)
        )
