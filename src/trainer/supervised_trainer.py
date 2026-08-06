from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
from loguru import logger
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.circuits.circuit import CircuitGenome


class SupervisedTrainer:
    """Trains EXAQC hybrid models using a fully batched execution path."""

    def __init__(
        self,
        training_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        training_loss_function: Callable[[Tensor, Tensor], Tensor],
        validation_loss_function: Callable[[Tensor, Tensor], Tensor],
        metrics: dict[str, Any],
        testing_dataloader: DataLoader | None = None,
        testing_loss_function: Callable[[Tensor, Tensor], Tensor] | None = None,
    ) -> None:
        """
        This creates a SupervisedTrainer object which can be (re)used to train circuit
        genomes given the provided training and validation dataloaders.

        Args:
            training_dataloader: a pytorch DataLoader object which can iterate over the
                training samples
            validation_dataloader: a pytorch DataLoader object which can iterate over the
                validation samples
            training_loss_function: provides the loss function used for training the
                genome
            validation_loss_function: provides the loss function used for calculating loss
                on the validation data. this may be different than the training data for
                example, when doing cross entropy loss where the class counts are different
                on training and validation data.
            metrics: is a dict where each key is the name of a metric, and each value is
                a function used to calculate a different metric (e.g., accuracy) which are
                different/in addition to the loss function.
            testing_dataloader: Optional held-out test dataloader.
            testing_loss_function: Optional held-out test loss function.
                Defaults to the validation loss function.
        """

        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.testing_dataloader = testing_dataloader
        self.training_loss_function = training_loss_function
        self.validation_loss_function = validation_loss_function
        self.testing_loss_function = (
            testing_loss_function
            if testing_loss_function is not None
            else validation_loss_function
        )
        self.metrics = metrics

    def get_metrics(
        self,
        genome: CircuitGenome,
        dataloader: DataLoader,
        loss_function: Callable[[Tensor, Tensor], Tensor],
        optimizer: Optimizer | None = None,
        epoch: int | None = None,
    ) -> dict[str, Any]:
        """
        Calculates the metrics for the provided genome and dataloader. Will optionally
            update gradients if is_training is set to true.

        Args:
            genome: the circuit genome to evaluate (without updating weights)
                on the validation data for this trainer.
            dataloader: a pytorch dataloader for the data to evaluate on.
            loss_function: the loss function to use for data evaluation.
            optimizer: is the optimizer use to train the genome if provided. if not
                provided metrics are just being gathered for inference/validation and
                weights should not be updated.
            epoch: the current epoch (if training), None otherwise. if specified this
                will be added to the metrics dict for better metadata parsing.

        Returns:
            A dictionary from each metric name to the metric value calculated
            over the validation data.
        """
        is_training = optimizer is not None
        genome.hybrid_model.train() if is_training else genome.hybrid_model.eval()

        for metric in self.metrics.values():
            metric.reset()

        total_loss = 0.0
        total_samples = 0

        with torch.set_grad_enabled(is_training):
            for batch_index, (x_batch, y_batch) in enumerate(dataloader):
                logger.debug("batch: {} / {}", batch_index, len(dataloader))

                if is_training:
                    optimizer.zero_grad(set_to_none=True)

                predictions = genome.forward(x_batch)

                if predictions.ndim != 2:
                    raise ValueError(
                        "Classification predictions must have shape "
                        "[batch_size, n_classes], received "
                        f"{tuple(predictions.shape)}."
                    )
                if predictions.shape[0] != y_batch.shape[0]:
                    raise ValueError(
                        "Prediction and target batch sizes differ: "
                        f"{predictions.shape[0]} != {y_batch.shape[0]}."
                    )

                loss = loss_function(predictions.float(), y_batch.long())

                if is_training:
                    loss.backward()
                    optimizer.step()

                current_batch_size = int(y_batch.shape[0])
                total_loss += float(loss.detach().item()) * current_batch_size
                total_samples += current_batch_size

                with torch.no_grad():
                    for prediction, target in zip(predictions, y_batch):
                        for metric in self.metrics.values():
                            metric.accumulate(prediction.float(), target.long())

        metric_results: dict[str, Any] = {
            "loss": (total_loss / total_samples if total_samples else float("nan"))
        }
        for metric_name, metric in self.metrics.items():
            metric_results[metric_name] = metric.calculate()

        if epoch is not None:
            metric_results["epoch"] = epoch
        return metric_results

    def train(self, genome: CircuitGenome) -> None:
        """
        Given the data loaders and loss functions provided to this SupervisedTrainer,
        this will train the provided circuit genome given its hyperparameter
        specifications.

        Args:
            genome: is the CircuitGenome to train. This method will initialize its
                self.circuit field so it can be trained with pytorch.
        """
        genome.initialize_model()

        hyperparameters = genome.hyperparameters
        learning_rate = float(hyperparameters["learning_rate"])
        epochs = int(hyperparameters["epochs"])

        # initalize the epoch metrics for tracking/data mining
        genome.metadata["training_epoch_metrics"] = []
        genome.metadata["validation_epoch_metrics"] = []

        n_trainable_parameters = sum(
            p.numel() for p in genome.hybrid_model.parameters() if p.requires_grad
        )

        logger.debug(f"hybrid model n trainable parameters: {n_trainable_parameters}")

        if n_trainable_parameters == 0:
            # this model has no parameters so it can't be trained. instead
            # just evaluate it on the validation data.
            logger.info("Model has no trainable parameters; evaluating only.")

            # calculate the metrics on the training data
            training_metric_results = self.get_metrics(
                genome,
                dataloader=self.training_dataloader,
                loss_function=self.training_loss_function,
            )

            # calculate the metrics on the validation data
            validation_metric_results = self.get_metrics(
                genome,
                dataloader=self.validation_dataloader,
                loss_function=self.validation_loss_function,
            )
            genome.metadata["best_training_metrics"] = training_metric_results
            genome.metadata["best_validation_metrics"] = validation_metric_results

            logger.info(f"training metrics were: {training_metric_results}")
            logger.info(f"validation metrics were: {validation_metric_results}")

            return

        optimizer = torch.optim.Adam(
            genome.hybrid_model.parameters(),
            lr=learning_rate,
            weight_decay=float(hyperparameters.get("weight_decay", 0.0)),
        )

        best_loss = math.inf
        best_epoch = 0
        improvement_cutoff = int(hyperparameters.get("improvement_cutoff", 2))
        best_parameters = {
            name: tensor.detach().clone()
            for name, tensor in genome.hybrid_model.state_dict().items()
        }

        for epoch in range(epochs):
            training_metric_results = self.get_metrics(
                genome,
                dataloader=self.training_dataloader,
                loss_function=self.training_loss_function,
                optimizer=optimizer,
                epoch=epoch,
            )
            logger.info(
                "[epoch {}] training metrics: {}",
                epoch,
                training_metric_results,
            )
            genome.metadata["training_epoch_metrics"].append(training_metric_results)

            # calculate the metrics on the validation data
            validation_metric_results = self.get_metrics(
                genome,
                dataloader=self.validation_dataloader,
                loss_function=self.validation_loss_function,
                epoch=epoch,
            )
            logger.info(
                "[epoch {}] validation metrics: {}",
                epoch,
                validation_metric_results,
            )
            genome.metadata["validation_epoch_metrics"].append(
                validation_metric_results
            )

            # TODO: try using the average of validation and training loss for fitness
            validation_loss = validation_metric_results["loss"]
            training_loss = training_metric_results["loss"]

            avg_loss = (validation_loss + training_loss) / 2.0

            if best_loss > avg_loss:
                best_loss = avg_loss
                best_epoch = epoch

                genome.metadata["best_training_metrics"] = training_metric_results
                genome.metadata["best_validation_metrics"] = validation_metric_results
                genome.metadata["best_epoch"] = best_epoch

                # get a copy of the current state dict of the hybrid model, this will be
                # all the weights
                with torch.no_grad():
                    best_parameters = {
                        name: tensor.detach().clone()
                        for name, tensor in (genome.hybrid_model.state_dict().items())
                    }
            elif epoch - best_epoch > improvement_cutoff:
                logger.info(
                    "Stopping at epoch {} because the last improvement "
                    "occurred at epoch {}.",
                    epoch,
                    best_epoch,
                )
                break

        logger.info(
            "Best loss found at epoch {} of {}.",
            best_epoch,
            epochs,
        )

        # set the genome's parameters to the ones from the best validation loss
        genome.set_parameters(best_parameters)

        return

    def test(
        self,
        genome: CircuitGenome,
    ) -> dict[str, Any]:
        """Evaluates a trained genome on the held-out test set.

        Args:
            genome: Trained genome to evaluate.

        Returns:
            Test loss and configured classification metrics.

        Raises:
            ValueError: If no test dataloader was provided.
        """
        if self.testing_dataloader is None:
            raise ValueError("No testing dataloader was provided to SupervisedTrainer.")

        test_metrics = self.get_metrics(
            genome=genome,
            dataloader=self.testing_dataloader,
            loss_function=self.testing_loss_function,
        )

        genome.metadata["testing_metrics"] = test_metrics

        return test_metrics
