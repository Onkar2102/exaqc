"""Tests for ``src.utils.helpers.draw_network`` across quantum circuits.

``draw_network`` renders a whole-model architecture image with ``visualtorch``.
Because a genome's hybrid model contains a quantum layer (a PennyLane
``TorchLayer`` or a qiskit ``TorchConnector``), visualtorch's tracer pushes a
``RecorderTensor`` through it, and the quantum math is dispatched via
``autoray``. ``draw_network`` aliases autoray's ``visualtorch`` backend to
``torch`` (once) and traces under ``no_grad`` so every required math operation
(``cos``, ``sin``, ``exp``, ``astype``, ...) resolves.

These tests verify that wiring end to end: for a variety of circuit
complexities on both the ``pennylane`` and ``qiskit`` targets, ``draw_network``
must actually **write** a valid layout PNG. Since ``draw_network`` catches its
own failures and merely logs a warning (writing nothing), asserting that the
file exists and is a valid PNG proves the trace genuinely succeeded rather than
having degraded to the skip path -- i.e. that no needed autoray op is missing.

All artifacts are written into pytest's per-test ``tmp_path`` (auto-removed),
so nothing is left behind in the repository.
"""

from __future__ import annotations

# Force a non-interactive matplotlib backend before anything imports pyplot,
# so any drawing runs headless and never opens a GUI window.
import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402

from src.utils.helpers import (  # noqa: E402
    draw_network,
    _register_visualtorch_autoray_backend,
)
from tests.supervised_trainer_test_utils import (  # noqa: E402
    build_classification_genome,
    COMPLEXITY_LEVELS_WITH_MULTI_PARAM,
)

#: Targets whose generated circuits are exercised.
TARGETS: tuple[str, ...] = ("pennylane", "qiskit")

#: The first eight bytes of any PNG file (the PNG signature).
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _build_initialized_genome(target: str, complexity: str):
    """Builds and initializes a classification genome ready for drawing.

    Reuses :func:`build_classification_genome` (the shared builder used across
    the trainer and save tests) so the circuit variety -- number of qubits,
    gate mix, and parametric gates per ``complexity`` -- is defined in one
    place.

    Args:
        target: Either ``"pennylane"`` or ``"qiskit"``.
        complexity: A complexity level understood by
            :func:`build_classification_genome`.

    Returns:
        A :class:`~src.circuits.circuit.CircuitGenome` whose ``hybrid_model``
        has been initialized.
    """
    genome, _ = build_classification_genome(
        genome_number=3,
        target=target,
        complexity=complexity,
        encoder_name="identity",
        decoder_name="clipped",
        include_parametric=True,
    )
    genome.initialize_model()
    return genome


@pytest.mark.parametrize("complexity", COMPLEXITY_LEVELS_WITH_MULTI_PARAM)
@pytest.mark.parametrize("target", TARGETS)
def test_draw_network_writes_valid_layout_png(
    target: str, complexity: str, tmp_path
) -> None:
    """``draw_network`` writes a valid layout PNG for each circuit and target.

    A successfully written PNG proves the visualtorch trace ran to completion,
    which in turn proves every autoray math operation the quantum layer needs
    is resolvable -- otherwise ``draw_network`` would have caught the error and
    written nothing.

    Args:
        target: Either ``"pennylane"`` or ``"qiskit"``.
        complexity: Circuit complexity level to build.
        tmp_path: pytest per-test temporary directory (auto-removed).
    """
    genome = _build_initialized_genome(target, complexity)

    draw_network(
        str(tmp_path),
        genome.hybrid_model,
        genome.genome_number,
        input_shape=genome.encoder.input_shape(),
    )

    layout_png = tmp_path / f"genome_{genome.genome_number}_layout.png"
    assert layout_png.is_file(), (
        f"draw_network did not write a layout PNG for target={target!r} "
        f"complexity={complexity!r}; the visualtorch trace likely failed "
        f"(a required autoray op may be missing)."
    )
    assert layout_png.stat().st_size > 0

    with open(layout_png, "rb") as handle:
        header = handle.read(len(_PNG_MAGIC))
    assert header == _PNG_MAGIC


@pytest.mark.parametrize("target", TARGETS)
def test_save_circuit_also_emits_layout_png(target: str, tmp_path, monkeypatch) -> None:
    """``save_circuit`` now emits the layout PNG alongside the circuit PNG.

    Exercises the real call path (``CircuitGenome.save_circuit`` invokes
    ``draw_network`` internally), confirming the integration produces the
    layout image.

    Args:
        target: Either ``"pennylane"`` or ``"qiskit"``.
        tmp_path: pytest per-test temporary directory (auto-removed).
        monkeypatch: used to ``chdir`` into ``tmp_path`` so a default out dir
            cannot escape into the repository.
    """
    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "artifacts"

    genome = _build_initialized_genome(target, "shallow")
    genome.metadata = {
        "best_training_metrics": {
            "loss": 0.1,
            "mean_class_accuracy": {"mean": 0.9},
        },
        "best_validation_metrics": {
            "loss": 0.2,
            "mean_class_accuracy": {"mean": 0.8},
        },
    }
    genome.save_circuit(insert_type="best", out_dir=str(out_dir))

    layout_png = out_dir / f"genome_{genome.genome_number}_layout.png"
    assert layout_png.is_file()
    with open(layout_png, "rb") as handle:
        assert handle.read(len(_PNG_MAGIC)) == _PNG_MAGIC


def test_autoray_resolves_required_math_ops_for_recorder_tensor() -> None:
    """After registration, autoray resolves torch math ops for the tracer.

    Directly asserts the mechanism the drawing tests rely on: visualtorch's
    ``RecorderTensor`` is treated as the ``torch`` backend, so the math
    operations a quantum layer dispatches through autoray resolve to their
    torch implementations instead of raising "couldn't find function ... for
    backend 'visualtorch'".
    """
    autoray = pytest.importorskip("autoray")
    recorder = pytest.importorskip("visualtorch.utils.recorder")
    import torch

    _register_visualtorch_autoray_backend()

    tracer = torch.zeros(2, 4).as_subclass(recorder.RecorderTensor)

    # The tracer's backend must now be torch, not visualtorch.
    assert autoray.infer_backend(tracer) == "torch"

    # Every op the quantum layers dispatch through autoray must resolve; a
    # missing one would raise ImportError here (and break draw_network).
    for op_name in ("cos", "sin", "exp", "astype"):
        if op_name == "astype":
            result = autoray.do("astype", tracer, "float64")
        else:
            result = autoray.do(op_name, tracer)
        assert result is not None
