import random
from typing import Any
from src.circuits.circuit import CircuitGenome


def gate_dropout(
    gates: list[Any],
    dropout_rate: float,
) -> set[int]:
    """Samples uniform gate dropout.

    Args:
        gates: Gates in the quantum circuit.
        dropout_rate: Probability of dropping each enabled gate.

    Returns:
        Innovation numbers of gates selected for dropout.
    """
    _validate_dropout_rate(dropout_rate)

    return {
        gate.innovation_number
        for gate in gates
        if gate.enabled and random.random() < dropout_rate
    }


def rotation_dropout(
    gates: list[Any],
    dropout_rate: float,
) -> set[int]:
    """Samples dropout over parameterized gates.
    
    Args:
        gates: Gates in the quantum circuit.
        dropout_rate: Probability of dropping each enabled gate.

    Returns:
        Innovation numbers of gates selected for dropout.
    """
    _validate_dropout_rate(dropout_rate)

    return {
        gate.innovation_number
        for gate in gates
        if (
            gate.enabled
            and len(gate.parameters) > 0
            and random.random() < dropout_rate
        )
    }


def entangling_dropout(
    gates: list[Any],
    dropout_rate: float,
) -> set[int]:
    """Samples dropout over multi-qubit gates.
    
    Args:
        gates: Gates in the quantum circuit.
        dropout_rate: Probability of dropping each enabled gate.

    Returns:
        Innovation numbers of gates selected for dropout.
    """
    _validate_dropout_rate(dropout_rate)

    return {
        gate.innovation_number
        for gate in gates
        if (
            gate.enabled
            and len(gate.qubits) > 1
            and random.random() < dropout_rate
        )
    }


def qubit_dropout(
    genome: CircuitGenome,
    dropout_rate: float,
) -> set[tuple[str, int]]:
    """Samples quantum dropout at the qubit level.

    Each candidate qubit is independently selected for dropout. Any
    evolved gate acting on a dropped qubit is skipped during the current
    forward pass.

    Input encoding and output measurement are not removed; only evolved
    gates are affected.
    
    Args:
        genome: The CircuitGenome.
        dropout_rate: Probability of dropping each enabled gate.

    Returns:
        Innovation numbers of gates selected for dropout.
    """
    _validate_dropout_rate(dropout_rate)

    return {
        qubit
        for qubit in genome.qubits
        if qubit not in genome.output_qubits and random.random() < dropout_rate
    }


def innovation_dropout(
    gates: list[Any],
    dropout_rate: float,
    innovation_strength: float = 0.5,
) -> set[int]:
    """Samples innovation-aware gate dropout.

    Newer innovations receive a larger dropout probability than
    older innovations.

    Args:
        gates: Gates in the quantum circuit.
        dropout_rate: Mean target dropout probability.
        innovation_strength: Strength of the innovation-rank bias.

    Returns:
        Innovation numbers of gates selected for dropout.
    """
    _validate_dropout_rate(dropout_rate)

    if not 0.0 <= innovation_strength <= 1.0:
        raise ValueError(
            "innovation_strength must be in [0, 1]."
        )

    enabled_gates = sorted(
        [gate for gate in gates if gate.enabled],
        key=lambda gate: gate.innovation_number,
    )

    n_gates = len(enabled_gates)

    if n_gates == 0:
        return set()

    dropped = set()

    for rank, gate in enumerate(enabled_gates):
        normalized_rank = (
            0.5
            if n_gates == 1
            else rank / (n_gates - 1)
        )

        # probability = dropout_rate * (
        #     1.0
        #     + innovation_strength
        #     * (2.0 * normalized_rank - 1.0)
        # )
        probability = dropout_rate * (
            0.5 + normalized_rank
        )

        probability = min(max(probability, 0.0), 1.0)

        if random.random() < probability:
            dropped.add(gate.innovation_number)

    return dropped


def _validate_dropout_rate(dropout_rate: float) -> None:
    """Validates a dropout probability."""
    if not 0.0 <= dropout_rate <= 1.0:
        raise ValueError(
            "dropout_rate must be in [0, 1], "
            f"received {dropout_rate}."
        )