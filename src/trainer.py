import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from src.config import Config
from src.model.model import Classifier
from src.utils.dataloader import get_dataloaders
from src.utils.logger import Logger
from src.utils.rng import RNG

PBAR_NCOLS = 92


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float, float]:
    """Calculate accuracy, precision, recall, and F1 score."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return accuracy, precision, recall, f1


class Trainer:
    """Trainer class for training and evaluating the model."""

    def __init__(self, config: Config, experiment_name: str | None = None) -> None:
        """Initialize the trainer."""
        self.config = config

        RNG.initialize(self.config.seed)

        self.model = Classifier(self.config.pretrained_model_name, self.config.classification_head)
        self.dataset = self.config.dataset(self.config.pretrained_model_name)

        if torch.cuda.is_available():
            self.device = "cuda:0"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate)

        self.train_dl, self.valid_dl, self.test_dl = get_dataloaders(
            self.dataset, config.train_test_split, config.batch_size
        )

        experiment_name = (
            experiment_name if experiment_name is not None else self.config.pretrained_model_name
        )
        self.logger = Logger(self.config, f"{experiment_name}")
        self.logger.log_metadata({"num_params": self.model.num_params})

        self.best_valid_f1: float | None = None
        self.best_epoch: int | None = None
        self.best_model_state_dict: dict[str, torch.Tensor] | None = None

    def train(self) -> None:
        """Train the model."""
        for epoch in range(self.config.num_train_epochs):
            self._train_epoch(epoch)
            valid_f1 = self._evaluate(epoch)

            # Store best model weights by validation F1
            if self.best_valid_f1 is None or valid_f1 > self.best_valid_f1:
                self.best_valid_f1 = valid_f1
                self.best_epoch = epoch
                self.best_model_state_dict = {
                    k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                }

        # Load best model at end of training
        if self.best_model_state_dict is not None:
            self.model.load_state_dict(self.best_model_state_dict, strict=True)
            self.model.to(self.device)
            self.logger.log_metadata(
                {"best_epoch": self.best_epoch, "best_valid_f1": self.best_valid_f1}
            )
            print(
                f"Loaded best model from epoch {self.best_epoch:2d} "
                f"(valid F1: {self.best_valid_f1:.4f})"
            )

    def test(self) -> None:
        """Evaluate the model on the test set."""
        self._evaluate()

    def save_model(self) -> None:
        """Save the trained model."""
        filename = f"{self.config.pretrained_model_name.replace('/', '-')}_{self.logger.id}.pt"
        saved_model_path = self.logger.model_dir / filename
        saved_model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), saved_model_path)
        self.logger.log_metadata({"saved_model": filename})

    def _train_epoch(self, epoch: int) -> None:
        """Run a single training epoch."""
        self.model.train()

        total_loss = 0.0
        all_preds, all_targets = [], []

        train_start = time.perf_counter()

        for batch in tqdm(
            self.train_dl, desc=f"Epoch {epoch:2d}".ljust(8), leave=False, ncols=PBAR_NCOLS
        ):
            x, y = batch
            y = y.to(self.device)
            x_input_id = x["input_ids"].squeeze(1).to(self.device)
            x_mask = x["attention_mask"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            y_pred = self.model(input_ids=x_input_id, attention_mask=x_mask)

            batch_loss = self.criterion(y_pred, y.long())
            batch_loss.backward()
            if self.config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

            self.optimizer.step()

            total_loss += batch_loss.item()

            # Collect predictions and targets for metrics
            with torch.no_grad():
                all_preds.extend(y_pred.argmax(-1).cpu().tolist())
                all_targets.extend(y.long().cpu().tolist())

        train_time_taken = time.perf_counter() - train_start

        avg_loss = total_loss / len(self.train_dl)
        accuracy, precision, recall, f1 = calculate_metrics(
            y_true=np.asarray(all_targets),
            y_pred=np.asarray(all_preds),
        )

        self.logger.log_train_result(
            {
                "epoch": epoch,
                "loss": avg_loss,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "train_time_taken": train_time_taken,
            }
        )

        print(
            f"Epoch {epoch:2d}".ljust(8) + " | "
            f"Loss: {avg_loss:.4f} | "
            f"Accuracy: {accuracy:.4f} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | "
            f"F1: {f1:.4f}"
        )

    def _evaluate(self, valid_epoch: int | None = None) -> float:
        """Run model evaluation.

        Evaluates on the validation set if valid_epoch is specified, otherwise on the test set.
        Returns the F1 score (for best-model tracking).
        """
        self.model.eval()

        total_loss = 0.0
        all_preds, all_targets = [], []

        desc = f"Valid {valid_epoch:2d}" if valid_epoch is not None else "Test"
        desc = desc.ljust(8)
        dataloader = self.valid_dl if valid_epoch is not None else self.test_dl

        evaluate_start = time.perf_counter()

        with torch.inference_mode():
            for batch in tqdm(dataloader, desc=desc, leave=False, ncols=PBAR_NCOLS):
                x, y = batch
                y = y.to(self.device)
                x_input_id = x["input_ids"].squeeze(1).to(self.device)
                x_mask = x["attention_mask"].to(self.device)

                y_pred = self.model(input_ids=x_input_id, attention_mask=x_mask)

                batch_loss = self.criterion(y_pred, y.long())
                total_loss += batch_loss.item()

                # Collect predictions and targets for metrics
                all_preds.extend(y_pred.argmax(-1).cpu().tolist())
                all_targets.extend(y.long().cpu().tolist())

        evaluate_time_taken = time.perf_counter() - evaluate_start

        avg_loss = total_loss / len(dataloader)
        accuracy, precision, recall, f1 = calculate_metrics(
            y_true=np.asarray(all_targets),
            y_pred=np.asarray(all_preds),
        )

        result_dict = {
            "epoch": valid_epoch,
            "loss": avg_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "evaluate_time_taken": evaluate_time_taken,
        }
        if valid_epoch is not None:
            self.logger.log_valid_result(result_dict)
        else:
            self.logger.log_test_result(result_dict)

        print(
            desc + " | "
            f"Loss: {avg_loss:.4f} | "
            f"Accuracy: {accuracy:.4f} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | "
            f"F1: {f1:.4f}"
        )

        return float(f1)
