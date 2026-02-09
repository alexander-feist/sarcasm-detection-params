from argparse import ArgumentParser, Namespace

from src.dataset.dataset_constructor import DatasetConstructor
from src.dataset.figlang import FigLangDataset
from src.dataset.isarcasm import ISarcasmDataset
from src.dataset.news_headlines import NewsHeadlinesDataset
from src.model.classification_heads import (
    BertCLSMLP,
    BertMeanPoolLinear,
    BertMultiHeadAttnPool,
    BertSingleTokenAttention,
    ClassificationHead,
)
from src.model.head_constructor import ClassificationHeadConstructor

DATASETS: dict[str, DatasetConstructor] = {
    "news_headlines": NewsHeadlinesDataset,
    "isarcasm": ISarcasmDataset,
    "figlang": FigLangDataset,
}

CLASSIFICATION_HEADS: dict = {
    "bert_mean_pool_linear": BertMeanPoolLinear,
    "bert_cls_mlp": BertCLSMLP,
    "bert_multi_head_attention": BertMultiHeadAttnPool,
    "bert_single_token_attention": BertSingleTokenAttention,
}


def get_dataset(name: str) -> DatasetConstructor:
    """Look up dataset constructor by key."""
    key = name.lower()
    if key not in DATASETS:
        options = ", ".join(sorted(DATASETS)) or "<none>"
        msg = f"Unknown dataset '{name}'. Available options: {options}."
        raise KeyError(msg)
    return DATASETS[key]


def get_classification_head(args: Namespace) -> ClassificationHeadConstructor:
    """Build a callable head constructor from args."""
    key = args.classification_head.lower()

    if key not in CLASSIFICATION_HEADS:
        options = ", ".join(sorted(CLASSIFICATION_HEADS)) or "<none>"
        msg = (
            f"Unknown classification head '{args.classification_head}'. "
            f"Available options: {options}."
        )
        raise KeyError(msg)

    head_cls = CLASSIFICATION_HEADS[key]

    kwargs: dict[str, object] = {"dropout_p": args.dropout_p}
    if head_cls is BertCLSMLP:
        kwargs["intermediate_size"] = args.intermediate_size
    if head_cls is BertMultiHeadAttnPool:
        kwargs["num_heads"] = args.num_heads

    def constructor(hidden_size: int) -> ClassificationHead:
        return head_cls(hidden_size=hidden_size, **kwargs)

    constructor.spec = {
        "head_type": head_cls.__name__,
        **dict(kwargs.items()),
    }

    return constructor


class ArgParser(ArgumentParser):
    """Argument parser."""

    def __init__(self) -> None:
        """Initialize the argument parser."""
        super().__init__()

        self.add_argument("--seed", type=int, default=1)
        self.add_argument("--dataset", default="news_headlines", choices=sorted(DATASETS))

        # Model
        self.add_argument("--pretrained-model-name", default="prajjwal1/bert-tiny")
        self.add_argument(
            "--classification-head",
            default="bert_single_token_attention",
            choices=sorted(CLASSIFICATION_HEADS),
        )
        self.add_argument("--dropout-p", type=float, default=0.1)
        self.add_argument(
            "--intermediate-size",
            type=int,
            default=None,
            help="Hidden size of MLP intermediate layer; uses 4 * head_hidden_size if None",
        )
        self.add_argument(
            "--num-heads",
            type=int,
            default=8,
            help="Number of attention heads for multi-head attention head",
        )

        # Training
        self.add_argument("--train-test-split", type=float, default=0.7)
        self.add_argument("--batch-size", type=int, default=32)
        self.add_argument("--learning-rate", type=float, default=2e-5)
        self.add_argument("--max-grad-norm", type=float, default=1.0)
        self.add_argument("--num-train-epochs", type=int, default=5)

        self.add_argument("--experiment-name", type=str, default=None)

        self.add_argument("--save-model", action="store_true")

    def parse(self) -> Namespace:
        """Parse command-line arguments."""
        return self.parse_args()
