import argparse
import json
from pathlib import Path

import torch
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.style import Style
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.reactive import reactive
from textual.widgets import Input, Label, Static
from transformers import BertTokenizer

from src.model.classification_heads import (
    BertCLSMLP,
    BertMeanPoolLinear,
    BertMultiHeadAttnPool,
    BertSingleTokenAttention,
    ClassificationHead,
)
from src.model.head_constructor import ClassificationHeadConstructor
from src.model.model import Classifier
from src.utils.utils import path_from_root

HEAD_TYPES: dict[str, type[ClassificationHead]] = {
    "BertMeanPoolLinear": BertMeanPoolLinear,
    "BertCLSMLP": BertCLSMLP,
    "BertMultiHeadAttnPool": BertMultiHeadAttnPool,
    "BertSingleTokenAttention": BertSingleTokenAttention,
}

LOGS_DIR = path_from_root("results/logs")


def get_experiment(path: Path) -> dict:
    """Get the experiment config by finding the log file from model ID."""
    model_id = path.stem.split("_")[-1]
    matches = list(LOGS_DIR.glob(f"**/{model_id}.json"))
    if not matches:
        msg = f"No log file found for model ID {model_id} in {LOGS_DIR}"
        raise FileNotFoundError(msg)
    return json.loads(matches[0].read_text(encoding="utf-8"))


def get_head_constructor(head_config: dict) -> ClassificationHeadConstructor:
    """Create a head constructor from experiment config."""
    head_type = head_config["head_type"]
    head_cls = HEAD_TYPES[head_type]

    kwargs: dict[str, object] = {"dropout_p": head_config.get("dropout_p", 0.1)}
    if head_type == "BertCLSMLP" and "intermediate_size" in head_config:
        kwargs["intermediate_size"] = head_config["intermediate_size"]
    if head_type == "BertMultiHeadAttnPool" and "num_heads" in head_config:
        kwargs["num_heads"] = head_config["num_heads"]

    def constructor(hidden_size: int) -> ClassificationHead:
        return head_cls(hidden_size=hidden_size, **kwargs)

    return constructor


def load_model(path: Path, device: str, experiment_log: dict) -> Classifier:
    """Load the model from the file path."""
    head_constructor = get_head_constructor(experiment_log["config"]["classification_head"])
    model = Classifier(experiment_log["config"]["pretrained_model_name"], head_constructor)

    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


class InferenceApp(App):
    """An app to run live inference."""

    CSS = """
    Screen {
        align: center middle;
    }

    #main-container {
        width: 80%;
        height: auto;
        border: round $accent-muted;
        padding: 1 2;
    }

    .label-title {
        margin-top: 1;
    }

    Input:focus {
        border: round $accent;
        background: transparent;
    }
    """

    # Reactive variable: when this changes, the UI updates automatically
    result_data = reactive(None)

    def __init__(self, path: Path, device: str) -> None:
        """Initialize the app."""
        super().__init__()
        self.experiment_path = path
        self.device = device
        self.experiment_log = get_experiment(path)
        self.base_model_name = self.experiment_log["config"]["pretrained_model_name"]
        self.tokenizer = BertTokenizer.from_pretrained(self.base_model_name)
        self.model = load_model(self.experiment_path, self.device, self.experiment_log)

    def compose(self) -> ComposeResult:
        """Compose the UI."""
        with Container(id="main-container"):
            yield Label("Input", classes="label-title")
            yield Input(placeholder="Type here...", id="text-input")

            yield Label("Prediction", classes="label-title")
            yield Static(
                Panel(Text("Type to predict..."), title="Prediction"),
                id="prediction-view",
            )

            yield Label("Saliency", classes="label-title")
            yield Static(
                Panel(
                    Text("Token saliency will appear here", style="dim"),
                    title="Token Impact",
                    border_style="dim",
                ),
                id="saliency-view",
            )

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input change."""
        self.result_data = self.get_prediction_and_saliency(event.value)

    def watch_result_data(self, data: tuple[list[float], list[str], list[float]] | None) -> None:
        """Watch for result data changes."""
        if data is None:
            return
        self.update_ui(*data)

    def get_prediction_and_saliency(self, text: str) -> tuple[list[float], list[str], list[float]]:
        """Get prediction and saliency for a given text.

        Returns:
            tuple[list[float], list[str], list[float]]: Tuple `(probs, tokens, scores)` where:

            - probs (list[float]): Class probabilities for this input.
            - tokens (list[str]): The tokens used.
            - scores (list[float]): Normalized importance score (0.0 to 1.0) per token.

        """
        if not text.strip():
            return [0.5, 0.5], [], []

        encoding = self.tokenizer(
            text,
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # Get embeddings (we need these to calculate gradients on them)
        embedding_layer = self.model.bert.embeddings.word_embeddings
        inputs_embeds = embedding_layer(input_ids).detach().requires_grad_()

        self.model.zero_grad(set_to_none=True)

        logits = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        pred_idx = int(probs.argmax(dim=1).item())
        log_probs[0, pred_idx].backward()

        # Extract saliency
        # Grads shape: (1, seq_len, hidden_dim)
        grads = inputs_embeds.grad[0]

        # Reduce hidden dim to get one score per token
        # This represents the "magnitude" of influence
        token_importance = torch.norm(grads, dim=1).detach().cpu().numpy()

        # Normalize to 0-1 range for coloring
        token_importance = token_importance / (token_importance.max() + 1e-7)

        # Get the actual string tokens for display
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        return probs[0].detach().cpu().tolist(), tokens, token_importance.tolist()

    def update_ui(self, probs: list[float], tokens: list[str], scores: list[float]) -> None:
        """Update the UI with the results."""
        prediction_view = self.query_one("#prediction-view", Static)
        saliency_view = self.query_one("#saliency-view", Static)

        color = "green" if probs[1] > probs[0] else "red"
        label = "SARCASTIC" if probs[1] > probs[0] else "NOT SARCASTIC"
        display_probs = max(probs[0], probs[1])
        high_importance_threshold = 0.5

        bar = ProgressBar(
            total=100,
            completed=display_probs * 100,
            width=None,
            style=f"dim {color}",
            complete_style=color,
        )
        prediction_view.update(
            Panel(bar, title=f"[b {color}]{label}[/] ({display_probs:.4f})", border_style=color)
        )

        rich_text = Text()

        for token, score in zip(tokens, scores, strict=True):
            if token in {"[CLS]", "[SEP]", "[PAD]"}:
                continue

            is_subword = token.startswith("##")
            display_token = token[2:] if is_subword else token
            prefix = "" if is_subword else " "

            r = int(100 + (155 * score))
            g = int(100 - (100 * score))
            b = int(100 - (100 * score))
            hex_color = f"#{r:02x}{g:02x}{b:02x}"

            style = Style(color=hex_color, bold=(score > high_importance_threshold))
            rich_text.append(prefix + display_token, style=style)

        saliency_view.update(Panel(rich_text, title="Token Impact", border_style="dim"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    app = InferenceApp(path=Path(args.path), device=args.device)
    app.run()
