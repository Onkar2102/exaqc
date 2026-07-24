import math
import torch

from loguru import logger

from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.circuits.circuit import CircuitGenome


class SupervisedTrainer:

    def __init__(
        self,
        training_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        training_loss_function: callable[[Tensor, Tensor], Tensor],
        validation_loss_function: callable[[Tensor, Tensor], Tensor],
        metrics: dict[str, callable[[Tensor, Tensor], Tensor]],
    ):
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
        """

        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.training_loss_function = training_loss_function
        self.validation_loss_function = validation_loss_function
        self.metrics = metrics

    def get_metrics(
        self,
        genome: CircuitGenome,
        dataloader: DataLoader,
        loss_function: callable[[Tensor, Tensor], Tensor],
        optimizer: Optimizer = None,
        epoch: int = None,
    ) -> dict[str, any]:
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

        with torch.set_grad_enabled(is_training):
            # set things up for calculating training metrics
            for metric_name, metric in self.metrics.items():
                metric.reset()

            loss_sum = Tensor([0])

            # x - inputs
            # y - targets
            # z - predictions
            n_batches = len(dataloader)
            batch = 0
            for x_batch, y_batch in dataloader:
                # logger.debug(f"batch: {batch} / {n_batches}")
                batch += 1
                batch_sum = Tensor([0])

                for x, y in zip(x_batch, y_batch):
                    z = genome.forward(x)

                    loss = loss_function(z.float(), y.long())
                    batch_sum += loss

                    # accumulate the metrics for each sample
                    with torch.no_grad():
                        for metric_name, metric in self.metrics.items():
                            metric.accumulate(z.float(), y.long())

                with torch.no_grad():
                    loss_sum += batch_sum

                # logger.debug(f"genome params:")
                # logger.debug(genome.hybrid_model.state_dict())

                if is_training:
                    batch_sum.backward()

                    optimizer.step()
                    optimizer.zero_grad()

            metric_results = {}
            metric_results["loss"] = loss_sum.detach().item()
            for metric_name, metric in self.metrics.items():
                metric_results[metric_name] = metric.calculate()

        if epoch is not None:
            metric_results["epoch"] = epoch

        return metric_results

    def train(
        self,
        genome: CircuitGenome,
    ):
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
        learning_rate = hyperparameters["learning_rate"]
        epochs = hyperparameters["epochs"]

        # initalize the epoch metrics for tracking/data mining
        genome.metadata["training_epoch_metrics"] = []
        genome.metadata["validation_epoch_metrics"] = []

        n_trainable_parameters = sum(p.numel() for p in genome.hybrid_model.parameters() if p.requires_grad)

        logger.debug(f"hybrid model n trainable parameters: {n_trainable_parameters}")

        if n_trainable_parameters == 0:
            # this model has no parameters so it can't be trained. instead
            # just evaluate it on the validation data.
            logger.info("model had no parameters so only evaluating the model on the data.")

            # calculate the metrics on the training data
            training_metric_results = self.get_metrics(genome, dataloader=self.training_dataloader, loss_function=self.training_loss_function)
            logger.info(f"training metrics were: {training_metric_results}")
            genome.metadata["best_training_metrics"] = training_metric_results

            # calculate the metrics on the validation data
            validation_metric_results = self.get_metrics(genome, dataloader=self.validation_dataloader, loss_function=self.validation_loss_function)
            logger.info(f"validation metrics were: {validation_metric_results}")
            genome.metadata["best_validation_metrics"] = validation_metric_results
            
            return

        optimizer = torch.optim.Adam(
            genome.hybrid_model.parameters(), lr=learning_rate, weight_decay=0.0
        )

        best_loss = math.inf
        epoch = 0
        best_epoch = 0
        best_parameters = {}
        improvement_cutoff = 2

        for epoch in range(epochs):
        # while True:
            training_metric_results = self.get_metrics(genome, dataloader=self.training_dataloader, loss_function=self.training_loss_function, optimizer=optimizer, epoch=epoch)
            logger.info(f"[epoch {epoch}] training metrics were: {training_metric_results}")
            genome.metadata["training_epoch_metrics"].append(training_metric_results)

            print("state dict:")
            print(optimizer.state_dict())
            print()

            # calculate the metrics on the validation data
            validation_metric_results = self.get_metrics(genome, dataloader=self.validation_dataloader, loss_function=self.validation_loss_function, epoch=epoch)
            logger.info(f"[epoch {epoch}] validation metrics were: {validation_metric_results}")
            genome.metadata["validation_epoch_metrics"].append(validation_metric_results)

            # TODO: try using the average of validation and training loss for fitness
            validation_loss = validation_metric_results['loss']
            training_loss = training_metric_results['loss']

            avg_loss = (validation_loss + training_loss) / 2.0

            if best_loss > avg_loss:
                genome.metadata["best_training_metrics"] = training_metric_results
                genome.metadata["best_validation_metrics"] = validation_metric_results

                best_loss = avg_loss
                best_epoch = epoch

                # get a copy of the current state dict of the hybrid model, this will be
                # all the weights
                with torch.no_grad():
                    best_parameters = { name : tensor.detach().clone() for name, tensor in genome.hybrid_model.state_dict().items() }

            elif (epoch - best_epoch) > improvement_cutoff:
                logger.info(f"stopping training on epoch {epoch} as last best epoch was {best_epoch} and no improvement found in {improvement_cutoff} epochs.")
                break

            epoch += 1

        logger.info(f"best loss found on epoch {best_epoch} of {epochs}")

        # set the genome's parameters to the ones from the best validation loss
        genome.set_parameters(best_parameters)

        return
