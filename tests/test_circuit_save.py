"""Tests for ``CircuitGenome.save_circuit`` output generation.

``save_circuit`` writes four artifacts into its ``out_dir`` for a genome:

* ``genome_<n>.json`` -- the serialized genome (round-trippable via
  ``CircuitGenome.from_dict``),
* ``genome_<n>.txt``  -- a human-readable gate listing,
* ``<insert_type>_genome_<n>_<tag>.png`` -- a drawing of the quantum circuit
  produced with the genome's target framework (pennylane or qiskit), and
* ``genome_<n>_layout.png`` -- a whole-model architecture diagram drawn by
  ``draw_network`` (see ``tests/test_draw_network.py`` for focused coverage).

All of these tests write exclusively into pytest's per-test ``tmp_path``
directory (which pytest creates and removes automatically) and additionally
``chdir`` into it, so a genome's default ``out_dir="artifacts/"`` cannot
create anything in the repository. Nothing is left behind on disk after a
run.
"""

from __future__ import annotations

# Force a non-interactive matplotlib backend *before* anything imports
# pyplot (src.circuits.circuit imports it at module load), so the drawing
# code runs headless and never tries to open a GUI window.
import matplotlib

matplotlib.use("Agg")

import json  # noqa: E402
import os  # noqa: E402
import pytest  # noqa: E402

from src.circuits.circuit import CircuitGenome  # noqa: E402

from tests.supervised_trainer_test_utils import (  # noqa: E402
    build_classification_genome,
)

TARGETS: tuple[str, ...] = ("pennylane", "qiskit")

#: The first eight bytes of any PNG file (the PNG signature).
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _build_saveable_genome(
    target: str,
    complexity: str = "shallow",
    *,
    with_metrics: bool = True,
    initialize: bool = True,
) -> CircuitGenome:
    """Builds a genome ready to be passed to ``save_circuit``.

    Args:
        target: Either ``"pennylane"`` or ``"qiskit"``.
        complexity: A circuit complexity level understood by
            ``build_classification_genome``.
        with_metrics: If True, populate ``metadata`` with the
            ``best_training_metrics``/``best_validation_metrics`` that
            ``save_circuit`` uses to build the PNG filename tag. If False,
            leave them out and instead set ``fitness`` so the fallback tag
            path is exercised.
        initialize: If True, call ``initialize_model()`` (the realistic
            post-training state). If False, leave the model uninitialized so
            the lazy circuit-generation branch of ``save_circuit`` is
            exercised.

    Returns:
        A configured :class:`CircuitGenome`.
    """

    genome, _ = build_classification_genome(
        genome_number=7,
        target=target,
        complexity=complexity,
        encoder_name="identity",
        decoder_name="clipped",
        include_parametric=True,
    )

    # Assign a fresh metadata dict (CircuitGenome's default argument is a
    # shared mutable dict) so tests stay isolated from one another.
    if with_metrics:
        genome.metadata = {
            "best_training_metrics": {
                "loss": 0.1234,
                "mean_class_accuracy": {"mean": 0.95},
            },
            "best_validation_metrics": {
                "loss": 0.2345,
                "mean_class_accuracy": {"mean": 0.85},
            },
        }
    else:
        genome.metadata = {}
        genome.fitness = {"train_return_mean": 1.5, "eval_return_mean": 2.5}

    if initialize:
        genome.initialize_model()

    return genome


def _split_by_suffix(directory: str) -> dict[str, list[str]]:
    """Groups the file names in ``directory`` by extension.

    Args:
        directory: Directory whose immediate entries should be grouped.

    Returns:
        A dict mapping each extension (e.g. ``".png"``) to the sorted list of
        file names with that extension.
    """

    grouped: dict[str, list[str]] = {}
    for name in sorted(os.listdir(directory)):
        grouped.setdefault(os.path.splitext(name)[1], []).append(name)
    return grouped


@pytest.mark.parametrize("complexity", ["shallow", "multi_param"])
@pytest.mark.parametrize("target", TARGETS)
def test_save_circuit_writes_exactly_the_expected_files(
    target: str, complexity: str, tmp_path, monkeypatch
) -> None:
    """``save_circuit`` writes exactly the expected artifacts and nothing else.

    The artifacts are the ``.json`` genome, the ``.txt`` gate listing, and two
    ``.png`` images: the circuit diagram (named from the metric tag) and the
    whole-model layout diagram (``genome_<n>_layout.png``).

    Args:
        target: Either ``"pennylane"`` or ``"qiskit"``.
        complexity: Circuit complexity level to build.
        tmp_path: pytest per-test temporary directory (auto-removed).
        monkeypatch: pytest fixture used to ``chdir`` into ``tmp_path`` so a
            default ``out_dir`` cannot escape into the repository.
    """

    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "artifacts"

    genome = _build_saveable_genome(target, complexity)
    genome.save_circuit(insert_type="best", out_dir=str(out_dir))

    by_suffix = _split_by_suffix(str(out_dir))

    # one json, one txt, and exactly two pngs (circuit + layout); nothing extra
    assert by_suffix.get(".json") == ["genome_7.json"]
    assert by_suffix.get(".txt") == ["genome_7.txt"]
    assert set(by_suffix) == {".json", ".txt", ".png"}

    png_names = by_suffix.get(".png", [])
    assert len(png_names) == 2

    # the circuit-diagram png name is built from insert_type, genome number,
    # and the metric tag; the layout diagram png has a fixed name.
    circuit_pngs = [name for name in png_names if name.startswith("best_genome_7_")]
    assert len(circuit_pngs) == 1
    circuit_png = circuit_pngs[0]
    assert "trainacc_0.9500" in circuit_png and "valacc_0.8500" in circuit_png
    assert "genome_7_layout.png" in png_names

    # every artifact has real content
    for name in ("genome_7.json", "genome_7.txt", *png_names):
        assert (out_dir / name).stat().st_size > 0


@pytest.mark.parametrize("target", TARGETS)
def test_save_circuit_png_is_a_valid_image(target: str, tmp_path, monkeypatch) -> None:
    """Every generated ``.png`` starts with the PNG signature bytes.

    Args:
        target: Either ``"pennylane"`` or ``"qiskit"``.
        tmp_path: pytest per-test temporary directory (auto-removed).
        monkeypatch: pytest fixture used to ``chdir`` into ``tmp_path``.
    """

    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "artifacts"

    genome = _build_saveable_genome(target)
    genome.save_circuit(insert_type="best", out_dir=str(out_dir))

    png_names = _split_by_suffix(str(out_dir))[".png"]
    assert png_names  # at least the circuit diagram
    for png_name in png_names:
        with open(out_dir / png_name, "rb") as handle:
            header = handle.read(len(_PNG_MAGIC))
        assert header == _PNG_MAGIC


@pytest.mark.parametrize("target", TARGETS)
def test_save_circuit_json_round_trips_via_from_dict(
    target: str, tmp_path, monkeypatch
) -> None:
    """The written JSON is valid and reconstructs an equivalent genome.

    Args:
        target: Either ``"pennylane"`` or ``"qiskit"``.
        tmp_path: pytest per-test temporary directory (auto-removed).
        monkeypatch: pytest fixture used to ``chdir`` into ``tmp_path``.
    """

    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "artifacts"

    genome = _build_saveable_genome(target)
    genome.save_circuit(insert_type="best", out_dir=str(out_dir))

    with open(out_dir / "genome_7.json") as handle:
        serialized = json.load(handle)

    restored = CircuitGenome.from_dict(serialized)
    assert restored.genome_number == genome.genome_number
    assert restored.target == genome.target
    assert restored.get_gate_innovations() == genome.get_gate_innovations()


@pytest.mark.parametrize("target", TARGETS)
def test_save_circuit_txt_lists_the_gates(target: str, tmp_path, monkeypatch) -> None:
    """The ``.txt`` artifact has the genome header and one line per gate.

    Args:
        target: Either ``"pennylane"`` or ``"qiskit"``.
        tmp_path: pytest per-test temporary directory (auto-removed).
        monkeypatch: pytest fixture used to ``chdir`` into ``tmp_path``.
    """

    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "artifacts"

    genome = _build_saveable_genome(target)
    genome.save_circuit(insert_type="best", out_dir=str(out_dir))

    contents = (out_dir / "genome_7.txt").read_text()
    assert "Genome 7" in contents
    for gate in genome.gates:
        if gate.enabled:
            assert gate.method_name in contents


@pytest.mark.parametrize("target", TARGETS)
def test_save_circuit_uses_fitness_fallback_tag_without_metrics(
    target: str, tmp_path, monkeypatch
) -> None:
    """Without training metrics, the PNG tag falls back to fitness values.

    Args:
        target: Either ``"pennylane"`` or ``"qiskit"``.
        tmp_path: pytest per-test temporary directory (auto-removed).
        monkeypatch: pytest fixture used to ``chdir`` into ``tmp_path``.
    """

    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "artifacts"

    genome = _build_saveable_genome(target, with_metrics=False)
    genome.save_circuit(insert_type="best", out_dir=str(out_dir))

    # The tag lives on the circuit-diagram png; the layout png has a fixed name.
    png_names = _split_by_suffix(str(out_dir))[".png"]
    circuit_png = next(name for name in png_names if not name.endswith("_layout.png"))
    assert "train_ret_1.5000" in circuit_png
    assert "val_ret_2.5000" in circuit_png


@pytest.mark.parametrize("target", TARGETS)
def test_save_circuit_works_without_initialize_model(
    target: str, tmp_path, monkeypatch
) -> None:
    """``save_circuit`` lazily generates the circuit when not yet initialized.

    Args:
        target: Either ``"pennylane"`` or ``"qiskit"``.
        tmp_path: pytest per-test temporary directory (auto-removed).
        monkeypatch: pytest fixture used to ``chdir`` into ``tmp_path``.
    """

    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "artifacts"

    genome = _build_saveable_genome(target, initialize=False)
    genome.save_circuit(insert_type="best", out_dir=str(out_dir))

    # the drawing branch must still have produced both pngs (circuit + layout),
    # not just json/txt, even though the genome was not pre-initialized
    png_names = _split_by_suffix(str(out_dir)).get(".png", [])
    assert len(png_names) == 2
    assert any(name.endswith("_layout.png") for name in png_names)


@pytest.mark.parametrize("target", TARGETS)
def test_save_circuit_only_touches_out_dir(target: str, tmp_path, monkeypatch) -> None:
    """Nothing is written outside the requested ``out_dir``.

    Runs from an otherwise empty working directory and confirms the only
    thing that appears in it is the requested output directory -- in
    particular no stray ``artifacts/`` directory from the default argument
    and no loose files.

    Args:
        target: Either ``"pennylane"`` or ``"qiskit"``.
        tmp_path: pytest per-test temporary directory (auto-removed).
        monkeypatch: pytest fixture used to ``chdir`` into ``tmp_path``.
    """

    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "nested" / "run_output"

    genome = _build_saveable_genome(target)
    genome.save_circuit(insert_type="best", out_dir=str(out_dir))

    # the working directory gained only the top-level "nested" directory
    assert sorted(os.listdir(tmp_path)) == ["nested"]
    assert os.listdir(tmp_path / "nested") == ["run_output"]
    # and the output directory holds only the three expected artifacts
    assert set(_split_by_suffix(str(out_dir))) == {".json", ".txt", ".png"}
