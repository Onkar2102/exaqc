"""Round-trip tests for ``CircuitGenome`` parameter get/set ordering.

These tests verify that :meth:`CircuitGenome.get_parameters_as_list` and
:meth:`CircuitGenome.set_parameters` read and write the quantum gate
parameters as the *same values in the same order*.

The two methods are the serialization boundary between the genome's gate
representation and the ``hybrid_model``'s flat quantum-weight tensor:

* ``get_parameters_as_list`` flattens every gate's parameters (in sorted-gate
  order) into the list used to initialize the quantum layer's weights.
* ``set_parameters`` takes a ``hybrid_model.state_dict()`` snapshot, extracts
  the quantum-layer weight tensor, and writes it back into ``gate.parameters``
  (via ``set_parameters_from_list``) in that same order.

If the two disagreed on ordering, trained weights would be silently permuted
across gates when written back into a genome, so this is worth pinning down
for both the ``pennylane`` and ``qiskit`` targets.
"""

from __future__ import annotations

import torch

import pytest

from src.circuits.circuit import CircuitGenome
from src.circuits.encoder import IdentityEncoder
from src.circuits.decoder import ClippedDecoder
from src.circuits.registers import expand_registers

from tests.supervised_trainer_test_utils import (
    COMPLEXITY_LEVELS_WITH_MULTI_PARAM,
    build_classification_genome,
    circuit_trainable_parameters,
    state_dict_with_quantum_weights,
)

TARGETS: tuple[str, ...] = ("pennylane", "qiskit")


def _gate_parameter_values_in_order(genome: CircuitGenome) -> list[float]:
    """Flattens gate parameters in gate order, independently of the methods under test.

    This mirrors the iteration ``get_parameters_as_list`` uses but is written
    out separately so a bug in that method cannot hide behind the same helper
    on both sides of an assertion.

    Args:
        genome: The genome whose gate parameters should be read.

    Returns:
        The gate parameter values as plain floats, in gate order.
    """

    values: list[float] = []
    for gate in genome.gates:
        for value in gate.parameters.values():
            values.append(float(value))
    return values


@pytest.mark.parametrize("complexity", COMPLEXITY_LEVELS_WITH_MULTI_PARAM)
@pytest.mark.parametrize("target", TARGETS)
def test_get_parameters_as_list_matches_quantum_weight_after_init(target: str, complexity: str) -> None:
    """After ``initialize_model``, the parameter list equals the quantum weight tensor.

    ``initialize_model`` seeds the quantum layer by copying
    ``get_parameters_as_list()`` into its weight tensor, so the two must agree
    element-for-element and in order. This anchors the "get" side of the
    round trip to the actual weight ordering the model trains on.

    Args:
        target: Either ``"pennylane"`` or ``"qiskit"``.
        complexity: One of the circuit complexity levels defined in
            ``tests.supervised_trainer_test_utils``.
    """

    genome, _ = build_classification_genome(
        genome_number=1,
        target=target,
        complexity=complexity,
        encoder_name="identity",
        decoder_name="clipped",
        include_parametric=True,
    )
    genome.initialize_model()

    parameter_list = genome.get_parameters_as_list()
    (quantum_weight,) = circuit_trainable_parameters(genome)

    assert len(parameter_list) == quantum_weight.numel()
    assert torch.allclose(
        torch.tensor([float(v) for v in parameter_list]),
        quantum_weight.detach().float(),
        atol=1e-6,
    )


@pytest.mark.parametrize("complexity", COMPLEXITY_LEVELS_WITH_MULTI_PARAM)
@pytest.mark.parametrize("target", TARGETS)
def test_set_parameters_writes_quantum_weights_back_in_order(target: str, complexity: str) -> None:
    """``set_parameters`` writes state-dict quantum weights into gates in order.

    Embeds a distinct, position-encoding weight vector (``[10, 20, 30, ...]``)
    into a hybrid-model state dict, applies it via ``set_parameters``, and
    checks that ``get_parameters_as_list`` -- and the raw gate parameters --
    come back as exactly that vector, in the same order. Distinct per-position
    values mean any transposition would be caught.

    Args:
        target: Either ``"pennylane"`` or ``"qiskit"``.
        complexity: One of the circuit complexity levels defined in
            ``tests.supervised_trainer_test_utils``.
    """

    genome, _ = build_classification_genome(
        genome_number=1,
        target=target,
        complexity=complexity,
        encoder_name="identity",
        decoder_name="clipped",
        include_parametric=True,
    )
    genome.initialize_model()

    n_parameters = len(genome.get_parameters_as_list())
    assert n_parameters >= 1

    new_weights = [10.0 * (i + 1) for i in range(n_parameters)]

    # a valid "best epoch" snapshot carrying the known weights
    state_dict = state_dict_with_quantum_weights(genome, new_weights)

    genome.set_parameters(state_dict)

    assert [float(v) for v in genome.get_parameters_as_list()] == new_weights
    # cross-check straight off the gates, not via get_parameters_as_list
    assert _gate_parameter_values_in_order(genome) == new_weights


@pytest.mark.parametrize("complexity", COMPLEXITY_LEVELS_WITH_MULTI_PARAM)
@pytest.mark.parametrize("target", TARGETS)
def test_get_then_set_same_weights_is_a_noop(target: str, complexity: str) -> None:
    """Feeding the current weights back through ``set_parameters`` changes nothing.

    Snapshots the genome's own (unchanged) weights and re-applies them; the
    gate parameters must be preserved in value and order. This is the identity
    case of the round trip and guards against accidental reordering or drift
    when the "best" epoch happens to be the current state.

    Comparison uses a tolerance because ``set_parameters`` reads the weights
    back out of the quantum layer's ``float32`` tensor (via ``.tolist()``), so
    values not exactly representable in ``float32`` pick up a tiny rounding
    delta -- an inherent property of the model, not a reordering.

    Args:
        target: Either ``"pennylane"`` or ``"qiskit"``.
        complexity: One of the circuit complexity levels defined in
            ``tests.supervised_trainer_test_utils``.
    """

    genome, _ = build_classification_genome(
        genome_number=1,
        target=target,
        complexity=complexity,
        encoder_name="identity",
        decoder_name="clipped",
        include_parametric=True,
    )
    genome.initialize_model()

    original = [float(v) for v in genome.get_parameters_as_list()]
    state_dict = state_dict_with_quantum_weights(genome, original)

    genome.set_parameters(state_dict)

    after = [float(v) for v in genome.get_parameters_as_list()]
    assert after == pytest.approx(original, abs=1e-6)


@pytest.mark.parametrize("target", TARGETS)
def test_parameter_order_follows_sorted_gate_depth(target: str) -> None:
    """Both get and set order parameters by gate depth, regardless of insert order.

    Builds a genome whose parametric gates are added out of depth order, each
    carrying a distinct sentinel value, then confirms:

    * ``get_parameters_as_list`` returns them sorted by gate depth (not
      insertion order); and
    * after ``set_parameters`` writes a distinct weight vector back, each gate
      -- identified by its depth -- receives the value at its sorted position.

    A single-qubit register with three ``rx`` rotations keeps the circuit
    trivially valid while isolating pure ordering behavior.

    Args:
        target: Either ``"pennylane"`` or ``"qiskit"``.
    """

    # rx's rotation parameter is named "theta" in qiskit but "phi" in pennylane.
    rx_parameter = "theta" if target == "qiskit" else "phi"

    genome = CircuitGenome(
        genome_number=1,
        target=target,
        input_qubits=expand_registers({"q": 1}),
    )
    genome.hyperparameters = {
        "learning_rate": 0.05,
        "epochs": 1,
        "quantum_input_mode": "u3",
        "quantum_output_mode": "probs",
    }

    # add gates out of depth order, each with a unique sentinel value
    genome.add_gate(depth=0.7, method_name="rx", qubits=[("q", 0)], parameters={rx_parameter: 0.77})
    genome.add_gate(depth=0.2, method_name="rx", qubits=[("q", 0)], parameters={rx_parameter: 0.22})
    genome.add_gate(depth=0.5, method_name="rx", qubits=[("q", 0)], parameters={rx_parameter: 0.55})

    # get must return values ordered by ascending gate depth, not insert order
    assert genome.get_parameters_as_list() == [0.22, 0.55, 0.77]

    n_inputs = genome.n_quantum_inputs()
    genome.encoder = IdentityEncoder(n_inputs=n_inputs, n_outputs=n_inputs)
    genome.decoder = ClippedDecoder(n_inputs=genome.n_quantum_outputs(), n_outputs=2)
    genome.initialize_model()

    # set a distinct weight vector and confirm each depth maps to its position
    state_dict = state_dict_with_quantum_weights(genome, [10.0, 20.0, 30.0])
    genome.set_parameters(state_dict)

    value_by_depth = {gate.depth: float(gate.parameters[rx_parameter]) for gate in genome.gates}
    assert value_by_depth == {0.2: 10.0, 0.5: 20.0, 0.7: 30.0}
    assert genome.get_parameters_as_list() == [10.0, 20.0, 30.0]


@pytest.mark.parametrize("target", TARGETS)
def test_multi_parameter_gate_preserves_within_gate_parameter_order(target: str) -> None:
    """A three-parameter ``u`` gate keeps its parameters in declared order.

    The gate-depth ordering test above only uses single-parameter gates, so
    it cannot catch a bug that permutes parameters *within* a single gate.
    This test builds a genome containing a ``u`` gate (qiskit ``u`` /
    pennylane ``U3``), which takes three trainable rotation parameters, and
    asserts that a distinct weight vector applied via ``set_parameters`` lands
    on the individual named parameters (``theta``, ``phi``, and the
    target-specific third name) in their declared order.

    The ``u`` gate's third parameter is ``"lam"`` in qiskit but ``"delta"`` in
    pennylane, so the expected parameter name is chosen per target.

    Args:
        target: Either ``"pennylane"`` or ``"qiskit"``.
    """

    # the u gate's third rotation parameter is named differently per target
    u_third_parameter = "lam" if target == "qiskit" else "delta"

    genome = CircuitGenome(
        genome_number=1,
        target=target,
        input_qubits=expand_registers({"q": 1}),
    )
    genome.hyperparameters = {
        "learning_rate": 0.05,
        "epochs": 1,
        "quantum_input_mode": "u3",
        "quantum_output_mode": "probs",
    }

    genome.add_gate(
        depth=0.5,
        method_name="u",
        qubits=[("q", 0)],
        parameters={"theta": 0.1, "phi": 0.2, u_third_parameter: 0.3},
    )

    # the flat list must follow the gate's parameter declaration order
    assert genome.get_parameters_as_list() == [0.1, 0.2, 0.3]

    n_inputs = genome.n_quantum_inputs()
    genome.encoder = IdentityEncoder(n_inputs=n_inputs, n_outputs=n_inputs)
    genome.decoder = ClippedDecoder(n_inputs=genome.n_quantum_outputs(), n_outputs=2)
    genome.initialize_model()

    # write distinct per-position values back and confirm each named
    # parameter receives the value at its position within the gate
    state_dict = state_dict_with_quantum_weights(genome, [10.0, 20.0, 30.0])
    genome.set_parameters(state_dict)

    (u_gate,) = genome.gates
    assert u_gate.parameters == {"theta": 10.0, "phi": 20.0, u_third_parameter: 30.0}
    # dict ordering (and therefore get_parameters_as_list) must be preserved
    assert list(u_gate.parameters.keys()) == ["theta", "phi", u_third_parameter]
    assert genome.get_parameters_as_list() == [10.0, 20.0, 30.0]
