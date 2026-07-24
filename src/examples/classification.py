from __future__ import annotations

import argparse
import math
import numpy as np
import os
import torch
import sys

from loguru import logger

from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from torch import Tensor
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.evolution.master_worker import master_worker

from src.evolution.steady_state_islands import SteadyStateIslands
from src.evolution.steady_state_population import SteadyStatePopulation
from src.evolution.objective import Objective
from src.circuits.pennylane_gate_specifications import pennylane_gate_specifications
from src.circuits.qiskit_gate_specifications import qiskit_gate_specifications
from src.circuits.circuit import (
    CircuitGenome,
    QUANTUM_INPUT_MODES,
    QUANTUM_OUTPUT_MODES,
)
from src.circuits.decoder import initialize_decoder, DECODING_OPTIONS
from src.circuits.encoder import initialize_encoder, ENCODING_OPTIONS

from src.datasets.classification import ClassificationDataset

from src.metrics.metric import Metric
from src.metrics.mean_class_accuracy import MeanClassAccuracy

from src.trainer.supervised_trainer import SupervisedTrainer


def get_dataset(dataset: str) -> (np.ndarray, np.ndarray):
    """
    Loads the specifed dataset and returns the samples and
    targets.

    Args:
        dataset: is name of the dataset to load.

    Returns:
        A tuple of (x, y) where x is the numpy array of
        sample values and y is a numpy array of the
        target classes.
    """

    x = None
    y = None

    if dataset in ["iris", "wine", "breast_cancer"]:
        data = None
        if dataset == "iris":
            data = load_iris()

        elif dataset == "wine":
            data = load_wine()

        elif dataset == "breast_cancer":
            data = load_breast_cancer()

        x = data.data
        y = data.target

    elif dataset == "seeds":
        data = np.loadtxt("src/datasets/classification/data/seeds_dataset.txt")
        # target_names = ["Kama", "Rosa", "Canadian"]

        x = data[:, :7]  # (210, 7)
        y = data[:, 7].astype(int)  # {1, 2, 3}

        # Convert labels -> {0, 1, 2}
        y = y - 1

    else:
        raise ValueError(dataset)

    return x, y


def get_dataloaders(
    x: np.ndarray,
    y: np.ndarray,
    normalize: str,
    training_size: float = 0.8,
    batch_size: int = 1,
    seed: int = 0,
) -> (DataLoader, DataLoader):
    """
    x: is a 2D numpy array with the values for each sample
    y: are the target labels (as ints)
    normalize: how to normalize the data, can currently be 'none', 'zscore' or 'minmax'
    training_size: is the percentage of the data to use for training
    seed: is the seed for the train_test_split method for reproducibility
    """

    if normalize not in ["none", "zscore", "minmax"]:
        logger.fatal(
            f"Specified normalization strategy '{normalize}' unknown or not supported."
        )
        exit(1)

    if normalize == "minmax":
        # for min max scaling we can just scale the whole dataset before the
        # split
        scaler = MinMaxScaler()
        x = scaler.fit_transform(x)

        # scale between 0 and pi instead of 0 and 1 because
        # weights are angles in the circuits
        x = x * math.pi

    x = x.astype(np.float32)

    x_train, x_validation, y_train, y_validation = train_test_split(
        x,
        y,
        train_size=training_size,
        random_state=seed,
        stratify=y,
    )

    logger.debug(f"x_train type: {x_train.dtype}")
    logger.debug(f"x_validation type: {x_validation.dtype}")
    logger.debug(f"y_train type: {y_train.dtype}")
    logger.debug(f"y_validation type: {x_validation.dtype}")

    if normalize == "zscore":
        # be careful with zscore normalization so that we normalize the
        # validation data with the same mean/stddev as the training data

        training_mean = np.mean(x_train, axis=0)
        training_std = np.std(x_train, axis=0)

        x_train = (x_train - training_mean) / (training_std + 1e-8)
        x_validation = (x_validation - training_mean) / (training_std + 1e-8)

    training_dataset = ClassificationDataset(x_train, y_train)
    validation_dataset = ClassificationDataset(x_validation, y_validation)

    # first get the count for each classes/labels in the dataset
    label_counts = torch.bincount(training_dataset.y)
    logger.debug(f"training label counts are: {label_counts}")

    # check to see if the label counts for the training samples
    # are different, i.e., it is class/label imbalanced
    if not ((label_counts == label_counts[0]).all().item()):
        # if so, create a weighted random sampler for the training dataloader

        # compute a weight for each label so that we get (on average)
        # evenly label-balanced batches
        label_weights = 1.0 / label_counts.float()
        logger.debug(f"training label weights are: {label_weights}")

        logger.info(
            f"Using a weighted random sampler with per class counts: {label_counts} and weights: {label_weights}"
        )

        # give each sample its own weight (which will be the same across
        # a class/label)
        sample_weights = label_weights[training_dataset.y]
        logger.debug(
            f"training sample weights (len {len(sample_weights)} are: {sample_weights}"
        )

        # TODO: probably better to do a custom rotator based dataloader so we can
        # guarantee that each batch has the same number of samples from each
        # class/label; but for now we can try pytorch's default method.
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,  # allow minority class(s) to be resampled/oversampled
        )

        # Create PyTorch DataLoaders for batching and shuffling
        training_loader = DataLoader(
            training_dataset, batch_size=batch_size, shuffle=False, sampler=sampler
        )
        training_loader.label_counts = label_counts
        training_loader.label_weights = label_weights

        training_loader.n_labels = len(label_weights)
        training_loader.n_features = len(x_train[0])

        """
        label_counts = [0, 0, 0]
        for i in range(1000):
            count = 0
            for samples, labels in training_loader:
                print(f"labels shape: {labels.shape} -- {labels}")
                for label in labels:
                    label_counts[label] += 1

                count += labels.shape[0]
            print(f"epoch count: {count}, label_counts: {label_counts}\n")
        """
    else:
        # otherwise just use a standard shuffled/batched dataloader
        logger.info("Using a standard shuffled/batched dataloader.")

        training_loader = DataLoader(
            training_dataset, batch_size=batch_size, shuffle=True
        )

    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False
    )

    validation_label_counts = torch.bincount(validation_dataset.y)
    logger.info(f"validation label counts are: {validation_label_counts}")

    validation_label_weights = 1.0 / validation_label_counts.float()
    logger.info(f"training label weights are: {validation_label_weights}")

    validation_loader.label_counts = validation_label_counts
    validation_loader.label_weights = validation_label_weights

    validation_loader.n_labels = len(validation_label_weights)
    validation_loader.n_features = len(x_validation[0])

    logger.info(
        f"number of training labels: {training_loader.n_labels} and features: {training_loader.n_features}"
    )
    logger.info(
        f"number of validation labels: {validation_loader.n_labels} and features: {validation_loader.n_features}"
    )

    assert training_loader.n_labels == validation_loader.n_labels
    assert training_loader.n_features == validation_loader.n_features

    return training_loader, validation_loader


# ---------------------------------------------------------------------
# Objective and single objective comparison
# ---------------------------------------------------------------------


def compare(genome1: CircuitGenome, genome2: CircuitGenome) -> int:
    """
    Used to sort genomes by fitness, even if there are multiple objectives, for population
    management and crossover methods.

    Returns: 0 if the two genomes have equivalent fitnesses, a ngeative value if genome1 should be
        sorted before genome2, and a positive value if genome2 should be sorted before genome1
    """

    # this will return 0 if the losses are the same, negative if genome1 should be before
    # genome2 (genome1's fitness would be lower), and positive if genome2 should be before
    # genome1 (genome2's fitness would be lower)
    return genome1.fitness["loss"] - genome2.fitness["loss"]


class ClassificationObjective(Objective):
    def __init__(
        self,
        training_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        training_loss_function: callable[[Tensor, Tensor], Tensor],
        validation_loss_function: callable[[Tensor, Tensor], Tensor],
        metrics: dict[str, Metric],
    ):
        self.trainer = SupervisedTrainer(
            training_dataloader=training_dataloader,
            validation_dataloader=validation_dataloader,
            training_loss_function=training_loss_function,
            validation_loss_function=validation_loss_function,
            metrics=metrics,
        )

    def __call__(self, genome: CircuitGenome):
        """
        Trains the circuit given the specified hyperparameters and sets its
        loss values.

        Args:
            genome: is the CircuitGenome to train and evaluate. It will have
                a dict of hyperparameters specified by EXAQC, and should
                set its `fitness` attribute with a dict of fitness values
                after training and evaluation.
        """

        self.trainer.train(genome)

        # set the loss (lower is better) and target metric (higher is better)
        # values for the genome fitness.
        genome.fitness = {
            "loss": (
                genome.metadata["best_validation_metrics"]["loss"]
                + genome.metadata["best_training_metrics"]["loss"]
            )
            / 2.0,
            "target_metric": (
                genome.metadata["best_validation_metrics"]["mean_class_accuracy"][
                    "mean"
                ]
                + genome.metadata["best_training_metrics"]["mean_class_accuracy"][
                    "mean"
                ]
            )
            / 2.0,
        }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset", choices=["iris", "wine", "seeds", "breast_cancer"], required=True
    )

    p.add_argument(
        "--out_dir",
        type=str,
        default="artifacts",
        help="Output directory to store results from runs",
    )

    p.add_argument(
        "--mutation_strategy",
        "-ms",
        type=str,
        nargs="+",
        required=True,
    )

    p.add_argument(
        "--parent_strategy",
        "-ps",
        type=str,
        nargs="+",
        required=True,
    )

    subparsers = p.add_subparsers(
        dest="population_strategy",
        help="Specify how genomes will be handled.",
        required=True,
    )

    steady_state_parser = subparsers.add_parser(
        "steady_state", help="Use a single steady state population."
    )
    steady_state_parser.add_argument("--max_population_size", type=int, default=30)

    islands_parser = subparsers.add_parser(
        "islands", help="Use multiple islands of steady state opulations."
    )
    islands_parser.add_argument("--n_islands", type=int, default=10)
    islands_parser.add_argument("--max_island_size", type=int, default=10)
    islands_parser.add_argument("--genomes_before_extinction", type=int, default=100)
    islands_parser.add_argument("--genomes_for_next_extinction", type=int, default=200)
    islands_parser.add_argument("--islands_to_extinct", type=int, default=2)
    islands_parser.add_argument("--primary_parent", type=str, default="best")
    islands_parser.add_argument(
        "--intra_island_crossover_rate", type=float, default=0.5
    )

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--learning_rate", "-lr", type=float, default=5e-4)
    p.add_argument("--number_genomes", type=int, default=2000)
    p.add_argument("--input_qubits", type=int)
    p.add_argument("--output_qubits", type=int)

    p.add_argument("--target", type=str, choices=["pennylane", "qiskit"])

    p.add_argument(
        "--quantum_input_mode",
        "-qim",
        choices=QUANTUM_INPUT_MODES,
        type=str,
        default="u3",
        help="Choose initial gate types whose parameteres will be  set from classical inputs",
    )

    p.add_argument(
        "--quantum_output_mode",
        "-qom",
        choices=QUANTUM_OUTPUT_MODES,
        type=str,
        default="probs",
        help="Choose the output mode from the quantum circuit.",
    )

    p.add_argument(
        "--encoding",
        choices=ENCODING_OPTIONS,
        type=str,
        default="linear",
        help="Choose the kind of encoding",
    )

    p.add_argument(
        "--decoding",
        choices=DECODING_OPTIONS,
        type=str,
        default="linear",
        help="Choose the kind of decoding",
    )

    p.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Use mini-batch training with the given batch size, if provided",
    )

    p.add_argument(
        "--logging_level",
        type=str,
        required=False,
        default="INFO",
        help="""One of the 5 default logging levels for showing on terminal. Pick DEBUG to show everything.""",
    )

    args = p.parse_args()

    # remove the old logging handler.
    logger.remove()
    # create a new logging handler at the appropriate level
    logger.add(sys.stdout, level=args.logging_level)
    logger.add(os.path.join(args.out_dir, "run.log"))

    logger.info(f"mutation strategy: {args.mutation_strategy}")

    # specify hyperparameter options for genome evaluation
    hyperparameters = {
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "quantum_input_mode": args.quantum_input_mode,
        "quantum_output_mode": args.quantum_output_mode,
    }

    x, y = get_dataset(args.dataset)

    training_dataloader, validation_dataloader = get_dataloaders(
        x=x,
        y=y,
        normalize="minmax",
        batch_size=hyperparameters["batch_size"],
    )

    metrics = {}
    metrics["mean_class_accuracy"] = MeanClassAccuracy(training_dataloader.n_labels)

    # set up the objective function
    objective = ClassificationObjective(
        training_dataloader=training_dataloader,
        validation_dataloader=validation_dataloader,
        training_loss_function=torch.nn.CrossEntropyLoss(
            weight=training_dataloader.label_weights
        ),
        validation_loss_function=torch.nn.CrossEntropyLoss(
            weight=validation_dataloader.label_weights
        ),
        metrics=metrics,
    )

    target = args.target

    # TODO: Determining the number of inputs/outputs could probably
    # be done in a nice wrapper function or something
    n_input_registers = args.input_qubits
    n_encoder_outputs = n_input_registers
    if args.quantum_input_mode == "u3":
        n_encoder_outputs *= 3

    n_output_registers = args.output_qubits
    n_decoder_inputs = n_output_registers

    if args.quantum_output_mode == "probs":
        n_decoder_inputs = 2**n_decoder_inputs

    initial_encoder = initialize_encoder(
        target=target,
        encoding_str=args.encoding,
        n_inputs=training_dataloader.n_features,
        n_outputs=n_encoder_outputs,
    )
    initial_decoder = initialize_decoder(
        target=target,
        decoding_str=args.decoding,
        n_inputs=n_decoder_inputs,
        n_outputs=training_dataloader.n_labels,
    )

    population = None

    if args.population_strategy == "steady_state":
        population = SteadyStatePopulation(
            max_population_size=args.max_population_size,
            compare=compare,
            out_dir=args.out_dir,
        )

    elif args.population_strategy == "islands":
        population = SteadyStateIslands(
            n_islands=args.n_islands,
            max_island_size=args.max_island_size,
            genomes_before_extinction=args.genomes_before_extinction,
            genomes_for_next_extinction=args.genomes_for_next_extinction,
            islands_to_extinct=args.islands_to_extinct,
            primary_parent=args.primary_parent,
            compare=compare,
            out_dir=args.out_dir,
        )

    gate_specifications = None
    if target == "pennylane":
        gate_specifications = pennylane_gate_specifications
    else:
        gate_specifications = qiskit_gate_specifications

    master_worker(
        gate_specifications=gate_specifications,
        population=population,
        objective=objective,
        initial_encoder=initial_encoder,
        initial_decoder=initial_decoder,
        hyperparameters=hyperparameters,
        mutation_strategy=args.mutation_strategy,
        parent_strategy=args.parent_strategy,
        run_for=args.number_genomes,
        input_registers={"input": n_input_registers},
        output_registers={"input": n_output_registers},
        target=target,
    )
