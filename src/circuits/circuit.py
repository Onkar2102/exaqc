from __future__ import annotations

import bisect
import os
import json
import matplotlib.pyplot as plt
import pennylane as qml
import torch


from loguru import logger
from typing import Dict, Optional

from qiskit import QuantumCircuit
from qiskit import QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector

from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


from src.circuits.gate import Gate
from src.circuits.decoder import Decoder
from src.circuits.encoder import Encoder
from src.objectives.genome_objectives import (
    genome_to_torch_params,
)

QUANTUM_INPUT_MODES = ["u3", "rx", "ry", "rz", "basis", "amplitude"]
QUANTUM_OUTPUT_MODES = ["probs", "expval", "state"]

class CircuitGenome:

    def __init__(
        self,
        genome_number: int,
        target: str,
        input_qubits: list[tuple[str, int]],
        output_qubits: list[tuple[str, int]] = None,
        metadata: dict[str, any] = {},
    ):
        """
        Initializes an empty quantum circuit.

        Args:
            genome_number: a unique identifier for this evolved circuit, which also represents the ordering
                that genomes have been generated, e.g., 0 is the first genome created, 1 is the next, etc.
            target: specifies if the circuit is for qiskit or pennylane
            input_qubits: a list of qubit names and indexes (e.g., (a, 0)).
            output_qubits: a list of qubit names and indexes (e.g., (a, 0)), if None then output_qubits are
                the same as the input qubits.
            metadata: is metadata about the genome used by things like the population strategy, etc. or to
                track other information about the genome.
        """
        self.genome_number = genome_number
        self.metadata = metadata

        # these should be specified by EXAQC

        # create a list of input qubits (which are tuples of register names and indexes)
        # so we can easily select random qubits to use for gate mutations
        self.qubits: list[tuple[str, int]] = []

        self.input_qubits: list[tuple[str, int]] = input_qubits
        for qubit in self.input_qubits:
            self.qubits.append(qubit)

        if output_qubits is None:
            output_qubits = input_qubits.copy()
        self.output_qubits: list[tuple[str, int]] = output_qubits
        for qubit in self.output_qubits:
            if qubit not in self.qubits:
                self.qubits.append(qubit)

        # make sure they are all sorted
        self.input_qubits.sort()
        self.output_qubits.sort()
        self.qubits.sort()

        # get indexes for input and output qubits in the full qubit list
        self.input_indexes = []
        for qubit in self.input_qubits:
            self.input_indexes.append(self.qubits.index(qubit))

        self.output_indexes = []
        for qubit in self.output_qubits:
            self.output_indexes.append(self.qubits.index(qubit))

        # a list of Gates sorted by depth represnting the gates in the quantum
        # circuit
        self.gates: list[Gate] = []

        self.target = target
        self.quantum_weight_key = "quantum_layer.weights" if self.target == "pennylane" else "quantum_layer.weight"

        # if a genome has not yet been evaluated, its fitness is None
        self.fitness = None

        # the torch parameters used for training  and the circuit for 
        # training are initially set to None and initialized if the
        # genome is used for training or validation
        self.torch_model = None
        self.torch_parameters = None


    def n_quantum_inputs(self) -> int:
        """
        Used to specify how many values the quantum circuit expects so
        encoders can be properly generated.

        Return:
            The number of inputs (the expected tensor size) for the quantum 
            circuit specified by this genome.
        """

        input_mode = self.hyperparameters["quantum_input_mode"]

        if input_mode == "u3":
            return len(self.input_indexes) * 3
        elif input_mode in ["rx", "ry", "rz"]:
            return len(self.input_indexes)
        else:
            raise ValueError(f"unknown quantum_input_mode={input_mode}")

    def n_quantum_outputs(self) -> int:
        """
        Used to specify how many values the quantum circuit will return
        so decoders can be properly generated.

        Return:
            The number of outputs (the expected tensor size) for the quantum 
            circuit specified by this genome.
        """

        output_mode = self.hyperparameters["quantum_output_mode"]

        if output_mode == "probs":
            return 2**len(self.output_indexes)

        elif output_mode == "expval":
            return len(self.output_indexes)

        else:
            raise ValueError(f"unknown quantum_output_mode={output_mode}")

 
    def is_valid(self) -> bool:
        """
        Returns:
            True if there is at least a path from the input qubits to the
            output qubits through the gates (i.e., the inputs can effect
            the outputs).  False, otherwise.
        """

        # determine which qubits this gate can be applied to so it will effect the output qubits
        self.sort_gates()

        reached_indexes = set(self.input_indexes)
        logger.debug(f"inital reached indexes now: {reached_indexes}")

        for gate in self.gates:
            if not gate.enabled:
                continue

            output_circuit_indexes = gate.get_output_circuit_indexes(self)
            input_circuit_indexes = gate.get_input_circuit_indexes(self)

            # if any of the input indexes for the gate are in the reached
            # qubit indexes, then this gate is effected by the input and we can add
            # its outputs as additional possible inputs

            if not set(input_circuit_indexes).isdisjoint(reached_indexes):
                reached_indexes.update(output_circuit_indexes)

            logger.debug(f"\treached indexes now: {reached_indexes}")

        valid = not reached_indexes.isdisjoint(self.output_indexes)
        logger.debug(
            f"output indexes are: {self.output_indexes}, circuit valid? {valid}"
        )

        return valid

    def dominates(self, other: CircuitGenome, loss: str = "loss") -> bool:
        """
        Determines if this genome dominates another genome. This method is needed because
        in the multi-objective case we can't just compare a single fitness value to determine
        if one genome is better than another.

        Args:
            other: is the other genome to compare to.

        Returns:
            True if this genome dominates another genome.
        """

        # TODO: update for multi objectives, but for now just use the given
        # loss key
        return self.fitness[loss] < other.fitness[loss]

    def get_gate_innovations(self) -> list[int]:
        """
        Returns:
            A sorted list of all the enabled gate innovation numbers in this
            genome.
        """
        gates = [gate.innovation_number for gate in self.gates if gate.enabled]
        gates.sort()
        return gates

    def has_same_gates(self, other: CircuitGenome) -> bool:
        """
        This checks to see if this genome has the exact same enabled
        gates as the other genome.

        Returns:
            True if both genomes have the same enabled gates innovation nubmers (but
            gates can potentially have different trained parameters).
        """

        self_gates = self.get_gate_innovations()
        other_gates = other.get_gate_innovations()

        logger.debug(
            f"comparing self gates {self_gates} to other gates {other_gates}, equal? {self_gates == other_gates}"
        )

        return self.get_gate_innovations() == other.get_gate_innovations()

    def copy(self, genome_number: int = None) -> CircuitGenome:
        """
        Creates a deep copy of this CircuitGenome, with potentially a new
        genome_number if it will be used as a child genome, e.g. for crossover
        or mutation.

        Args:
            genome_number: if this is specified, the copy will use this new genome
                number. This also means the fitness should be set to None as it will
                be modified via crossover or mutation.

        Returns:
            A copy of this genome, with potentially modified genome number and fitness.
        """

        fitness = self.fitness

        if genome_number is None:
            genome_number = self.genome_number
            fitness = None

        new_genome = CircuitGenome(
            genome_number=genome_number,
            target=self.target,
            input_qubits=self.input_qubits.copy(),
            output_qubits=self.output_qubits.copy(),
        )
        new_genome.metadata = self.metadata.copy()
        new_genome.fitness = fitness
        new_genome.hyperparameters = self.hyperparameters.copy()

        for gate in self.gates:
            new_genome.add_existing_gate(gate)

        return new_genome

    def to_dict(self) -> dict(str, any):
        """
        Creates a dict representation of the circuit genome that can be converted to JSON
        or used for MPI serialization. This won't contain any of the qiskit or pennylane
        internals which will need to be recreated when it is loaded back with the
        CircuitGenome.from_dict method.

        Returns:
            A simple dict representation of this CircuitGenome.
        """

        serialized = {}
        serialized["fitness"] = self.fitness
        serialized["genome_number"] = self.genome_number
        serialized["metadata"] = self.metadata
        serialized["target"] = self.target
        serialized["input_qubits"] = self.input_qubits.copy()
        serialized["output_qubits"] = self.output_qubits.copy()
        serialized["hyperparameters"] = self.hyperparameters.copy()
        serialized["gates"] = []

        serialized["encoder"] = self.encoder.to_dict()
        serialized["decoder"] = self.decoder.to_dict()

        for gate in self.gates:
            serialized["gates"].append(gate.to_dict())

        return serialized

    @classmethod
    def from_dict(cls, serialized: dict[str, any]) -> CircuitGenome:
        """
        Args:
            serialized: is a serialized version of a CircuitGenome created
                by the to_dict method.

        Returns:
            A circuit genome created from a serialized dict of a circuit genome.
        """
        new_genome = CircuitGenome(
            genome_number=serialized["genome_number"],
            target=serialized["target"],
            input_qubits=serialized["input_qubits"],
            output_qubits=serialized["output_qubits"],
            metadata=serialized["metadata"],
        )
        new_genome.fitness = serialized["fitness"]
        new_genome.hyperparameters = serialized["hyperparameters"]

        new_genome.encoder = Encoder.from_dict(serialized["encoder"])
        new_genome.decoder = Decoder.from_dict(serialized["decoder"])

        for serialized_gate in serialized["gates"]:
            gate = Gate.from_dict(serialized_gate)
            new_genome.add_existing_gate(gate)

        return new_genome

    def add_existing_gate(self, gate: Gate):
        """
        Adds a new already created gate to this quantum circuit, keeping the
        gates in order sorted first by depth and then by innovation number to
        handle any gates with the same depth (which shouldn't usually happen).

        Args:
            gate: is the gate to add.
        """

        bisect.insort(self.gates, gate, key=lambda g: (g.depth, g.innovation_number))

    def add_gate(
        self,
        depth: float,
        method_name: str,
        qubits: list[tuple[str, int]] = [],
        parameters: dict[str, float] = {},
        innovation_number: int = None,
    ):
        """
        Adds a new already created gate to this quantum circuit, keeping the
        gates in order sorted first by depth and then by innovation number to
        handle any gates with the same depth (which shouldn't usually happen).

        Args:
            depth: a number between 0 and 1 representing the depth of the gate in the circuit.
            method_name: the name of the method to invoke this gate on a qiskit QuantumCircuit
            qubits: a list of qubits to form the arguments to the gate method name, each is a tuple
                with a string for the input register name and then the index.
            parameters: a dict where the key is the parameter name and the value is the parameter value
        """

        gate = Gate(
            depth=depth,
            method_name=method_name,
            qubits=qubits,
            parameters=parameters,
            target=self.target,
        )
        # make sure to add the gate in sorted order
        bisect.insort(self.gates, gate, key=lambda g: (g.depth, g.innovation_number))

    def sort_gates(self):
        """
        Sorts the gates in the circuit by their depth (useful if new gates are
        added or the circuit is mutated).

        Sort the gates first by depth then by innovation number (in case two gates
        somehow had the same depth).
        """
        self.gates.sort(key=lambda g: (g.depth, g.innovation_number))

    def get_possible_input_qubits(self, depth: float) -> list[int]:
        """
        Traces back the gates from the input to a given depth to determine which input
        qubits will effect any of the final output gates that are being measured.

        Args:
            depth: the depth a new gate will be added at, the results of this method will be
                used to determine which qubits can be used as input (control) parameters for
                the gate.

        Returns:
            A list of potential qubit indexes in this circuit that will effect the output
            qubits.
        """
        # determine which qubits this gate can be applied to so it will effect the output qubits
        self.sort_gates()

        possible_input_indexes = set(self.input_indexes)

        for gate in self.gates:
            if not gate.enabled:
                continue

            output_circuit_indexes = gate.get_output_circuit_indexes(self)
            input_circuit_indexes = gate.get_input_circuit_indexes(self)

            # if any of the input indexes for the gate are in the possible input
            # qubit indexes, then this gate is effected by the input and we can add
            # its outputs as additional possible inputs

            if not set(input_circuit_indexes).isdisjoint(possible_input_indexes):
                possible_input_indexes.update(output_circuit_indexes)

            if gate.depth >= depth:
                # we've gone through all gates ahead of the insertion
                # depth for this new gate.
                break

            if len(possible_input_indexes) == len(self.qubits):
                # all gates are possible so we can quit checking
                break

        return sorted(possible_input_indexes)

    def get_possible_output_qubits(self, depth: float) -> list[int]:
        """
        Traces back the gates from the output to a given depth to determine which output
        qubits will effect any of the final output gates that are being measured.

        Args:
            depth: the depth a new gate will be added at, the results of this method will be
                used to determine which qubits can be used as output (target) parameters for
                the gate.

        Returns:
            A list of potential qubit indexes in this circuit that will effect the output
            qubits.
        """
        # determine which qubits this gate can be applied to so it will effect the output qubits
        reverse_gates = sorted(
            self.gates, key=lambda g: (g.depth, g.innovation_number), reverse=True
        )

        possible_output_indexes = set(self.output_indexes)

        for gate in reverse_gates:
            if not gate.enabled:
                continue

            output_circuit_indexes = gate.get_output_circuit_indexes(self)
            input_circuit_indexes = gate.get_input_circuit_indexes(self)

            # if any of the output indexes for the gate are in the possible output
            # qubit indexes, then this gate effects the output and we can add its
            # inputs as effecting the output
            if not set(output_circuit_indexes).isdisjoint(possible_output_indexes):
                possible_output_indexes.update(input_circuit_indexes)

            if gate.depth <= depth:
                # we've gone through all gates ahead of the insertion
                # depth for this new gate.
                break

            if len(possible_output_indexes) == len(self.qubits):
                # all gates are possible so we can quit checking
                break

        return sorted(possible_output_indexes)

    def get_parameters_as_list(self) -> list[float]:
        """
        This is used to get the all the gate parameters as a list (in order
        of the sorted gates) so that they can be tracked by the qiskit
        and pennylane models.

        Returns:
            All the parameter values in this quantum circuit as a list of
            float values.
        """
        parameter_list = []

        for gate in self.gates:
            # set each gate and its parameters using the weight vector
            for parameter_name, value in gate.parameters.items():
                parameter_list.append(value)

        return parameter_list

    def set_parameters_from_list(self, parameter_list: list[float | Tensor]):
        """
        This takes a list of parameters, in the same order as those generated
        by the `get_parameters_as_list` method and sets the float values of
        all the gate parameters from them.

        Args:
            parameter_list: is a list of floating point or tensor values which
                will be used to set the gate parameter values.
        """

        offset = 0
        for gate in self.gates:
            # set each gate and its parameters using the weight vector
            for parameter_name, value in gate.parameters.items():
                gate.parameters[parameter_name] = parameter_list[offset]
                offset += 1

        return parameter_list


    def set_parameters(self, state_dict: dict[str, Tensor]):
        """
        Sets the parameters of the circuit genome after a trainer has finished its
        epochs over it. These should be set from the parameters from the best validation
        loss, which will be floats (not Tensors).
        """

        with torch.no_grad():
            # copy all tensors from the saved state dict to the current
            # state dict (this will handle encoders/decoders)
            current_state_dict = self.hybrid_model.state_dict()

            for name, tensor in state_dict.items():
                current_state_dict[name].copy_(tensor)

            # get the quantum parameter list so we can use this to set
            # the CircuitGenome gate parameters

            # of course pennylane and qiskit use a slightly different name for weights
            quantum_parameter_list = None
            if self.target == "pennylane":
                quantum_parameter_list = current_state_dict["quantum_layer.weights"].tolist()
            else:
                quantum_parameter_list = current_state_dict["quantum_layer.weight"].tolist()

            print(f"quantum parameter list: {quantum_parameter_list}")
            self.set_parameters_from_list(quantum_parameter_list)

    def initialize_model(self):
        """
        Will generate an appropriate pennylane or qiskit circuit for this
        circuit genome, given specified hyperparameters (which should be
        initialized with the genome). After this is completed, self.torch_model
        should be set to the model, which can take a tensor of input and
        calculate the outputs (which is wrapped in the forward method of
        CircuitGenome).
        """

        if self.target == "pennylane":
            self.generate_pennylane_circuit()
        elif self.target == "qiskit":
            self.generate_qiskit_circuit()
        else:
            logger.fatal(
                f"Unknown target {self.target} for circuit genome model generation."
            )
            exit(1)

        target = self.target
        n_quantum_inputs = self.n_quantum_inputs()
        n_quantum_outputs = self.n_quantum_outputs()

        parameter_list = self.get_parameters_as_list()
        torch_parameters = torch.Tensor(parameter_list)

        class HybridModel(torch.nn.Module):
            def __init__(self, encoder: Encoder, quantum_layer : TorchConnector | TorchLayer, decoder: Decoder):
                super().__init__()
                self.encoder = encoder
                self.quantum_layer = quantum_layer
                # Classical post-processing layer: expands 4 quantum probabilities to 3 classes
                self.decoder = decoder

            def forward(self, x: Tensor):
                x = self.encoder(x, self)
                # make sure the encoder is properly specified
                assert len(x) == n_quantum_inputs

                x = self.quantum_layer(x)

                # make sure the circuit outputs are properly specified
                assert len(x) == n_quantum_outputs

                x = self.decoder(x, self)

                return x

        self.hybrid_model = HybridModel(self.encoder, self.torch_model, self.decoder)


        with torch.no_grad():
            # initialize the torch parameters
            # and of course the weight name is different
            if self.target == "pennylane":
                print(f"hybrid_model.quantum_layer.weights (before copy): {self.hybrid_model.quantum_layer.weights}")
                self.hybrid_model.quantum_layer.weights.copy_(torch.tensor(parameter_list))
                print(f"hybrid_model.quantum_layer.weight (after copy): {self.hybrid_model.quantum_layer.weights}")

            else:
                print(f"hybrid_model.quantum_layer.weight (before copy): {self.hybrid_model.quantum_layer.weight}")
                self.hybrid_model.quantum_layer.weight.copy_(torch.tensor(parameter_list))
                print(f"hybrid_model.quantum_layer.weight (after copy): {self.hybrid_model.quantum_layer.weight}")

    def forward(self,
        x: Tensor,
    ) -> torch.Tensor:
        """
        Does a forward pass through the `self.torch_model` model initialized
        in the :meth:``initialize_model` method.

        Args:
            x: is the input sample batch to pass through the model
            params: 
        """

        print(f"doing forward pass, encoder.n_inputs: {self.encoder.n_inputs}, encoder.n_outputs: {self.encoder.n_outputs}, quantum_inputs: {self.n_quantum_inputs()}, quantum_outputs: {self.n_quantum_outputs()}, decoder.n_inputs: {self.decoder.n_inputs}, decoder.n_outputs: {self.decoder.n_outputs}")


        return self.hybrid_model(x)


    def generate_pennylane_circuit(
        self,
        device_name: str = "default.qubit",
        measure_registers: bool = False,
        shots: Optional[int] = None,
        return_probs: bool = False,
    ):
        """
        Converts this genome into a PennyLane QNode-ready function.

        Args:
            device_name: Name of the PennyLane device to use.
            measure_registers: If True, return measurement results for all wires (like Qiskit classical registers).
            shots: Sample from circuit output,
            return_probs: Return the actual probablity weights from the circuit

        Returns:
            A tuple (dev, qnode_fn), where `dev` is the PennyLane device and
            `qnode_fn` is a QNode function that implements this circuit genome.
        """
        # Create wire registers via qml.registers
        self.total_qubits = len(self.qubits)

        logger.info(
            f"input indexes: {self.input_indexes}, output_indexes: {self.output_indexes}"
        )

        # Instantiate PennyLane device
        dev = qml.device(
            device_name,
            wires=self.total_qubits,
            shots=shots,
        )

        output_mode = self.hyperparameters["quantum_output_mode"]
        logger.info(f"output mode is: {output_mode}")

        # Define the QNode function
        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def qnode_fn(
            inputs: torch.Tensor,
            weights: torch.Tensor,
        ):

            # initialize the qubits given the specified input mode
            input_mode = self.hyperparameters["quantum_input_mode"]

            if input_mode == "u3":
                for i, w in enumerate(self.input_indexes):
                    start = i * 3
                    qml.U3(inputs[start], inputs[start+1], inputs[start+2], w)

            elif input_mode == "rx":
                for i, w in enumerate(self.input_indexes):
                    qml.RX(inputs[i], w)

            elif input_mode == "ry":
                for i, w in enumerate(self.input_indexes):
                    qml.RY(inputs[i], w)

            elif input_mode == "rz":
                for i, w in enumerate(self.input_indexes):
                    qml.RZ(inputs[i], w)
            
            elif input_mode == "basis":
                qml.BasisState(inputs, wires=genome.input_indexes)

            elif input_mode == "amplitude":
                qml.AmplitudeEmbedding(
                    features=inputs,
                    wires=genome.input_indexes,
                    normalize=True,
                    pad_with=0.0,
                )

            else:
                raise ValueError(f"Unknown quantum_input_mode={quantum_input_mode}")

            # Apply all gates in depth order
            self.sort_gates()
            offset = 0
            for gate in self.gates:
                gate.add_to_pennylane_circuit(self.qubits, weights=weights, offset=offset)
                offset += len(gate.parameters)

            if output_mode == "probs":
                return qml.probs(wires=self.output_indexes)

            elif output_mode == "expval":
                expvals = [
                    qml.expval(qml.PauliZ(w))
                    for w in self.output_indexes  # self.register_map["output"]
                ]
                return expvals

            elif output_mode == "state":
                return qml.state()

            else:
                raise ValueError(f"Unknown quantum_output_mode={quantum_output_mode}")

        # set up the qiskit weights ParameterVector
        parameter_list = self.get_parameters_as_list()

        weight_shapes = { "weights": (len(parameter_list),) }

        self.torch_model = qml.qnn.TorchLayer(qnode_fn, weight_shapes)


    def generate_qiskit_circuit(self):
        """
        Converts this genome into a useable qiskit instationation.

        Returns:
            A qiskit QuantumCircuit instantiation of this circuit genome.
        """
        quantum_registers = []
        classical_registers = []
        register_dict = {}
        for qubit_name, qubit_index in self.qubits:
            quantum_register = QuantumRegister(1, name=f"{qubit_name}-{qubit_index}")
            quantum_registers.append(quantum_register)
            register_dict[(qubit_name, qubit_index)] = quantum_register

        classical_register = ClassicalRegister(len(self.output_qubits), name="c")
        circuit = QuantumCircuit(*quantum_registers, classical_register)

        '''
        for qubit_name, qubit_index in self.output_qubits:
            classical_registers.append(ClassicalRegister(1))

        circuit = QuantumCircuit(*quantum_registers, *classical_registers)
        '''


        # initialize the qubits given the specified input mode
        input_mode = self.hyperparameters["quantum_input_mode"]
        inputs = ParameterVector("x", length=self.n_quantum_inputs())

        if input_mode == "u3":
            for i, w in enumerate(self.input_indexes):
                start = i * 3
                circuit.u(inputs[start], inputs[start+1], inputs[start+2], w)

        elif input_mode == "rx":
            for i, w in enumerate(self.input_indexes):
                circuit.rx(inputs[i], w)

        elif input_mode == "ry":
            for i, w in enumerate(self.input_indexes):
                circuit.ry(inputs[i], w)

        elif input_mode == "rz":
            for i, w in enumerate(self.input_indexes):
                circuit.rx(inputs[i], w)

        else:
            raise ValueError(f"Unknown quantum_input_mode={quantum_input_mode}")

        # make sure we apply the gates in the correct ordering by depth
        self.sort_gates()

        # set up the qiskit weights ParameterVector
        parameter_list = self.get_parameters_as_list()

        # set up the weight vector used for tracking and training qiskit
        # gate weights
        self.weight_vector = ParameterVector("weights", length=len(parameter_list))

        offset = 0
        for gate in self.gates:
            # set each gate and its parameters using the weight vector
            gate.add_to_qiskit_circuit(register_dict, circuit, self.weight_vector, offset)
            offset += len(gate.parameters)


        for output_index, input_index in enumerate(self.output_indexes):
            print(f"measuring quantum_register[{input_index}] to classical_register[{output_index}]")
            circuit.measure(
                quantum_registers[input_index], classical_register[output_index]
            )
            '''
            circuit.measure(
                quantum_registers[input_index], classical_registers[output_index]
            )
            '''

        # determine which type of QNN to utilize to get the appropriate
        # outputs
        output_mode = self.hyperparameters["quantum_output_mode"]

        if output_mode == "probs":
            pm = generate_preset_pass_manager(optimization_level=1)  # good default
            estimator = StatevectorEstimator()
            gradient = ParamShiftEstimatorGradient(estimator=estimator)

            print(f"sampler num_clbits: {circuit.num_clbits}")

            # a hack to get the sampler to return the right number of qubits for the classical
            # output registers
            def identity_interpret(x: int) -> int:
                return x

            qnn = SamplerQNN(
                circuit=circuit,
                input_params=inputs,
                weight_params=self.weight_vector,
                input_gradients=True,
                interpret=identity_interpret,
                output_shape=2**circuit.num_clbits,
            )

            self.torch_model = TorchConnector(qnn)
            print(f"parameter_list: {parameter_list}")
            print(f"torch_model.weight: {self.torch_model.weight}")


        elif output_mode == "expval":
            self.torch_model = None

        else:
            raise ValueError(f"Unknown quantum_output_mode={output_mode}")


    def save_circuit(
        self,
        insert_type: str,
        out_dir: str = "artifacts/",
    ):
        """
        Saves this genome into the specified output directory in JSON and
        txt format.

        Args:
            insert_type: a tag to put at the beginning of the filename, e.g.
                'best' for global_best genomes.
            out_dir: where to write the genome files.
        """
        os.makedirs(out_dir, exist_ok=True)

        json_path = os.path.join(out_dir, f"genome_{self.genome_number}.json")
        logger.info(f"writing NEW BEST gnome to {json_path}")
        with open(json_path, "w") as fp:
            json.dump(self.to_dict(), fp, ensure_ascii=False, indent=4)

        # --- Text gate list ---
        txt_path = os.path.join(out_dir, f"genome_{self.genome_number}.txt")
        with open(txt_path, "w") as f:
            self.sort_gates()
            f.write(f"Genome {self.genome_number}\n")
            f.write(f"Qubits: {self.qubits}\n\n")
            for g in self.gates:
                if getattr(g, "enabled", True):
                    f.write(
                        f"{g.depth:.3f}  {g.method_name}  {g.qubits}  {g.parameters}\n"
                    )

        # --- PennyLane draw ---
        self.generate_pennylane_circuit(return_probs=True)

        print("metadata:")
        print(self.metadata)

        try:
            training_loss = self.metadata['best_training_metrics']['loss']
            training_accuracy = self.metadata['best_training_metrics']['mean_class_accuracy']['mean']

            validation_loss = self.metadata['best_validation_metrics']['loss']
            validation_accuracy = self.metadata['best_validation_metrics']['mean_class_accuracy']['mean']

            tag = (
                f"trainloss_{training_loss:.4f}_trainacc_{training_accuracy:.4f}_valloss_"
                f"{validation_loss:.4f}_valacc_{validation_accuracy:.4f}"
            )
        except Exception:
            tag = (
                f"best_ep_return_{self.fitness['best_episode_return']:.4f}_"
                f"eval_return_mean_{self.fitness['eval_return_mean']:.4f}"
            )

        try:
            params = genome_to_torch_params(self)
            x0 = torch.zeros(self.encoder.n_inputs)
            fig, ax = qml.draw_mpl(self.torch_model)(x0, params)
            ax.set_title(f"Genome {self.genome_number}")
            path = os.path.join(
                out_dir, f"{insert_type}_genome_{self.genome_number}_{tag}.png"
            )
            fig.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Could not draw circuit: {e}")
