from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import torch
import visualtorch
import os

from collections import deque
from typing import Optional
from loguru import logger

if TYPE_CHECKING:
    from src.circuits.circuit import CircuitGenome


GATE_COMPLEXITY = {
    "id": {"gate_count": 1, "cnot_count": 0, "rot_count": 0},
    "h": {"gate_count": 1, "cnot_count": 0, "rot_count": 0},
    "x": {"gate_count": 1, "cnot_count": 0, "rot_count": 0},
    "y": {"gate_count": 1, "cnot_count": 0, "rot_count": 0},
    "z": {"gate_count": 1, "cnot_count": 0, "rot_count": 0},
    "s": {"gate_count": 1, "cnot_count": 0, "rot_count": 0},
    "sdg": {"gate_count": 1, "cnot_count": 0, "rot_count": 0},
    "t": {"gate_count": 1, "cnot_count": 0, "rot_count": 0},
    "tdg": {"gate_count": 1, "cnot_count": 0, "rot_count": 0},
    "sx": {"gate_count": 1, "cnot_count": 0, "rot_count": 0},
    "sxdg": {"gate_count": 1, "cnot_count": 0, "rot_count": 0},
    "rx": {"gate_count": 1, "cnot_count": 0, "rot_count": 1},
    "ry": {"gate_count": 1, "cnot_count": 0, "rot_count": 1},
    "rz": {"gate_count": 1, "cnot_count": 0, "rot_count": 1},
    "p": {"gate_count": 1, "cnot_count": 0, "rot_count": 1},
    "u": {"gate_count": 1, "cnot_count": 0, "rot_count": 1},
    "u3": {"gate_count": 1, "cnot_count": 0, "rot_count": 1},
    "r": {"gate_count": 1, "cnot_count": 0, "rot_count": 1},
    "rv": {"gate_count": 1, "cnot_count": 0, "rot_count": 1},
    "cx": {"gate_count": 1, "cnot_count": 1, "rot_count": 0},
    "cy": {"gate_count": 1, "cnot_count": 1, "rot_count": 0},
    "cz": {"gate_count": 1, "cnot_count": 1, "rot_count": 0},
    "cp": {"gate_count": 1, "cnot_count": 2, "rot_count": 3},
    "crx": {"gate_count": 1, "cnot_count": 2, "rot_count": 2},
    "cry": {"gate_count": 1, "cnot_count": 2, "rot_count": 2},
    "crz": {"gate_count": 1, "cnot_count": 2, "rot_count": 2},
    "swap": {"gate_count": 1, "cnot_count": 3, "rot_count": 0},
    "iswap": {"gate_count": 1, "cnot_count": 0, "rot_count": 0},
    "rxx": {"gate_count": 1, "cnot_count": 2, "rot_count": 1},
    "ryy": {"gate_count": 1, "cnot_count": 2, "rot_count": 1},
    "rzz": {"gate_count": 1, "cnot_count": 2, "rot_count": 1},
    "rzx": {"gate_count": 1, "cnot_count": 2, "rot_count": 1},
    "ccx": {"gate_count": 1, "cnot_count": 0, "rot_count": 0},
    "ccz": {"gate_count": 1, "cnot_count": 0, "rot_count": 0},
    "ch": {"gate_count": 1, "cnot_count": 0, "rot_count": 0},
    "cswap": {"gate_count": 1, "cnot_count": 0, "rot_count": 0},
    "mcx": {"gate_count": 1, "cnot_count": 0, "rot_count": 0},
    "cs": {"gate_count": 1, "cnot_count": 2, "rot_count": 3},
    "csdg": {"gate_count": 1, "cnot_count": 2, "rot_count": 3},
    "csx": {"gate_count": 1, "cnot_count": 0, "rot_count": 0},
    "dcx": {"gate_count": 1, "cnot_count": 2, "rot_count": 0},
    "ecr": {"gate_count": 1, "cnot_count": 4, "rot_count": 2},
    "rccx": {"gate_count": 1, "cnot_count": 3, "rot_count": 0},
    "rcccx": {"gate_count": 1, "cnot_count": 8, "rot_count": 0},
    "ms": {"gate_count": 1, "cnot_count": 2, "rot_count": 5},
    "mcp": {"gate_count": 1, "cnot_count": 0, "rot_count": 1},
    "mcrx": {"gate_count": 1, "cnot_count": 0, "rot_count": 1},
    "mcry": {"gate_count": 1, "cnot_count": 0, "rot_count": 1},
    "mcrz": {"gate_count": 1, "cnot_count": 0, "rot_count": 1},
    "cu": {"gate_count": 1, "cnot_count": 2, "rot_count": 6},
}


def register_wire_map(registers: dict[str, int]) -> dict:
    """Return a dict mapping register names to PennyLane wires."""
    wire_map = {}
    offset = 0
    for name, size in registers.items():
        wire_map[name] = list(range(offset, offset + size))
        offset += size
    return wire_map


class BalancedBatchSampler:
    """Draws mini-batches with equal class representation.

    Each class maintains its own independent queue of indices.  When a
    class queue is exhausted it is refilled (and optionally shuffled)
    before sampling continues.  Because class sizes differ, queues empty
    at different rates, so shuffles are triggered independently per class.

    Args:
        data:             List of (features, y_onehot, cls_name) tuples.
        batch_size:       Total samples per batch.  Rounded down to the
                          nearest multiple of the number of classes.
        shuffle:          Whether to shuffle a class bucket when it is
                          exhausted and refilled.
    """

    def __init__(self, data: list, batch_size: Optional[int], shuffle: bool) -> None:
        self.data = data
        self.shuffle = shuffle
        self.n_samples = len(data)

        self.rng = np.random.default_rng(seed=42)

        class_indices: dict[str, list[int]] = {}
        for i, (_, _, cls) in enumerate(data):
            class_indices.setdefault(cls, []).append(i)

        self.classes: list[str] = sorted(class_indices.keys())
        self.num_classes: int = len(self.classes)
        self.class_indices = class_indices

        logger.debug(f"created a balanced batch sampler with classes: {self.classes}")
        logger.debug(f"data size: {len(data)}")
        logger.debug(f"num_classes: {self.num_classes}")
        logger.debug("class_sizes:")
        for cls, indices in class_indices.items():
            logger.debug(f"\t'{cls}': {len(indices)}")

        if batch_size < self.num_classes:
            logger.error(
                f"ERROR: batch size {batch_size} must be at least the number of classes "
                "in the dataset ({self.num_classes}) for the balanced class sampler."
            )
            exit(1)

        self.samples_per_class = batch_size // self.num_classes

        logger.debug(f"samples_per_class: {self.samples_per_class}")

        self.batch_size = self.samples_per_class * self.num_classes
        logger.debug(f"batch_size: {self.batch_size}")

        # One deque per class — these persist across sample() calls
        self._queues: dict[str, deque[int]] = {
            cls: self._make_queue(cls) for cls in self.classes
        }

    def _make_queue(self, cls: str) -> deque[int]:
        """Build a fresh index queue for one class, shuffling if enabled.

        Args:
            cls: The class name whose index queue should be (re)built.

        Returns:
            A deque of indices into ``self.data`` for the given class,
            optionally shuffled.
        """
        indices = list(self.class_indices[cls])
        if self.shuffle:
            np.random.shuffle(indices)
        return deque(indices)

    def _draw(self, cls: str, n: int) -> list[int]:
        """Pull n indices from a class queue, refilling eagerly on exhaustion.

        When the last index is consumed the queue is immediately refilled so
        that ``len(self._queues[cls])`` always reflects items remaining in the
        current cycle rather than an ambiguous empty state.

        Args:
            cls: The class name to draw from.
            n: Number of indices to draw.

        Returns:
            List of n indices into ``self.data``.
        """
        out: list[int] = []
        queue = self._queues[cls]
        while len(out) < n:
            if not queue:
                queue.extend(self._make_queue(cls))
            out.append(queue.popleft())
        if not queue:
            queue.extend(self._make_queue(cls))
        return out

    def sample(self) -> list:
        """Return one balanced batch, advancing each class queue independently.

        Returns:
            List of ``self.batch_size`` (features, y_onehot, cls_name) tuples
            with exactly ``self.samples_per_class`` entries per class.
        """
        batch = []
        for cls in self.classes:
            indices = self._draw(cls, self.samples_per_class)
            batch.extend(self.data[i] for i in indices)
        return batch

    def sample_random(self):
        """Return a random sample from the dataset"""
        idx = self.rng.integers(0, self.n, size=self.batch_size)
        return [self.data[i] for i in idx.tolist()]

    def reset(self) -> None:
        """Rebuild all class queues from scratch.

        Resets the sampler to its initial state, as if no samples had been
        drawn. Useful when starting a new epoch with a fresh ordering.
        """
        self._queues = {cls: self._make_queue(cls) for cls in self.classes}


def genome_to_torch_params(genome: CircuitGenome) -> dict[str, torch.nn.Parameter]:
    """Extract trainable genome parameters into torch Parameters.

    Iterates over enabled gates in the genome and converts each gate parameter
    into a `torch.nn.Parameter`. Parameters are keyed using the stable identifier
    `<innovation_number>:<parameter_name>`.

    Args:
        genome (CircuitGenome): Quantum circuit genome with parametric gates.

    Returns:
        dict[str, torch.nn.Parameter]: Mapping from parameter keys to torch Parameters.
    """
    params: dict[str, torch.nn.Parameter] = {}
    for gate in genome.gates:
        if gate.enabled:
            for name, value in gate.parameters.items():
                key = f"{gate.innovation_number}:{name}"
                params[key] = torch.nn.Parameter(
                    torch.tensor(float(value), dtype=torch.float64)
                )
    return params


def _extract_param_value(v: torch.Tensor | float) -> float:
    """Convert a tensor or scalar parameter to a Python float.

    Args:
        v (torch.Tensor | float): Parameter value.

    Returns:
        float: Extracted scalar value.
    """
    if isinstance(v, torch.Tensor):
        return float(v.detach().cpu().item())
    return float(v)


def torch_params_to_genome(
    genome: CircuitGenome, trained_params: dict[str, torch.Tensor] | dict[str, float]
):
    """Write trained torch parameters back into a genome.

    Parameters are matched using `<innovation_number>:<parameter_name>` keys.

    Args:
        genome (CircuitGenome): Genome to update.
        trained_params (dict[str, torch.Tensor | float]): Trained parameters.
    """
    for gate in genome.gates:
        if gate.enabled:
            for name in gate.parameters.keys():
                key = f"{gate.innovation_number}:{name}"
                if key in trained_params:
                    gate.parameters[name] = _extract_param_value(trained_params[key])


#: Set once :func:`_register_visualtorch_autoray_backend` has run, so the
#: autoray backend alias is installed at most once per process.
_VISUALTORCH_AUTORAY_REGISTERED = False


def _register_visualtorch_autoray_backend() -> None:
    """Make ``autoray`` treat visualtorch's tracer tensors as torch tensors.

    ``visualtorch`` traces a model by pushing a ``RecorderTensor`` (a
    ``torch.Tensor`` subclass whose module is ``visualtorch``) through
    ``forward``. Quantum layers dispatch their math through ``autoray``, which
    infers a tensor's backend from its top-level module name -- here
    ``"visualtorch"`` -- and then fails with e.g. "couldn't find function 'cos'
    for backend 'visualtorch'", because no math implementations are registered
    for that backend.

    Aliasing the ``"visualtorch"`` backend to ``"torch"`` makes autoray resolve
    *every* required operation (``cos``, ``sin``, ``exp``, ``astype``, ``real``,
    ``imag``, ...) to its torch implementation at once -- correct here because a
    ``RecorderTensor`` is a ``torch.Tensor`` and torch functions operate on it
    (and are recorded via its ``__torch_function__``). This is far more robust
    than hand-registering individual functions, which would silently miss ops
    needed by gate or input-mode combinations not yet exercised.

    Returns:
        None. Installs the alias in autoray's backend-alias table and clears its
        cached backend inference so the alias takes effect immediately. Runs at
        most once per process and never raises: a missing or incompatible
        autoray is logged and ignored, since the layout image is best-effort.
    """
    global _VISUALTORCH_AUTORAY_REGISTERED
    if _VISUALTORCH_AUTORAY_REGISTERED:
        return
    _VISUALTORCH_AUTORAY_REGISTERED = True

    try:
        import autoray.autoray as autoray_internals

        autoray_internals._BACKEND_ALIASES["visualtorch"] = "torch"

        # Backend inference is memoized per tensor class; drop any cached
        # result so a RecorderTensor seen before the alias still maps to torch.
        for cache_name in (
            "_infer_class_backend_cached",
            "_infer_class_backend_multi_cached",
        ):
            cached = getattr(autoray_internals, cache_name, None)
            if cached is not None and hasattr(cached, "cache_clear"):
                cached.cache_clear()
    except Exception as error:
        logger.warning(
            "Could not alias the visualtorch autoray backend to torch; "
            "network layout drawing may fail: {}",
            error,
        )


def draw_network(
    out_dir: str,
    model: torch.nn.Module,
    genome_number: int,
    input_shape: tuple[int, ...] | None = None,
) -> None:
    """Draw an end-to-end-architecture view of the entire model.

    The installed ``visualtorch`` (>= 1.x) traces a model by pushing a tensor of
    ``input_shape`` through its ``forward`` and now *requires* ``input_shape`` as
    an argument. To let a hybrid model containing a quantum layer be traced, the
    tracer's math is routed through torch via
    :func:`_register_visualtorch_autoray_backend`, and the trace is run under
    ``torch.no_grad`` (the simulator otherwise tries to call ``numpy()`` on
    gradient-tracking tensors). Any remaining tracing failure is logged and the
    layout image is skipped rather than raised.

    Args:
        out_dir: The file path to the save directory.
        model: The model to visualize (e.g. the genome's hybrid model).
        genome_number: The genome identifier, used in the output filename.
        input_shape: The shape of a single input batch expected by ``model``,
            including the batch dimension (e.g. ``(1, n_features)`` for a linear
            encoder or ``(1, channels, height, width)`` for a CNN encoder). If
            ``None``, no shape can be inferred and the layout image is skipped.

    Returns:
        None. Writes ``genome_<n>_layout.png`` into ``out_dir`` on success, or
        logs a warning and returns without writing on failure.
    """
    if input_shape is None:
        logger.warning(
            "Skipping network layout image for genome {}: no input_shape "
            "was provided to draw_network.",
            genome_number,
        )
        return

    _register_visualtorch_autoray_backend()

    try:
        # no_grad: the traced statevector simulation would otherwise call
        # numpy() on gradient-tracking tensors and raise.
        with torch.no_grad():
            visualtorch.layered_view(
                model,
                input_shape=input_shape,
                legend=True,
                to_file=os.path.join(out_dir, f"genome_{genome_number}_layout.png"),
            )
    except Exception as error:
        # The layout diagram is best-effort: if visualtorch still cannot trace
        # a particular model, degrade to a concise warning instead of aborting
        # save_circuit.
        logger.warning(
            "Skipping network layout image for genome {}: visualtorch could "
            "not trace the model ({}).",
            genome_number,
            error,
        )
