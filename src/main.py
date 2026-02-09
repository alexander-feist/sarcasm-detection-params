from src.config import Config
from src.trainer import Trainer
from src.utils.arg_parser import ArgParser, get_classification_head, get_dataset


def main() -> None:
    """Run the main entry point."""
    parser = ArgParser()
    args = parser.parse_args()

    config = Config(
        seed=args.seed,
        dataset=get_dataset(args.dataset),
        # Model
        pretrained_model_name=args.pretrained_model_name,
        classification_head=get_classification_head(args),
        # Training
        train_test_split=args.train_test_split,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
    )

    trainer = Trainer(config=config, experiment_name=args.experiment_name)
    trainer.train()
    trainer.test()
    if args.save_model:
        trainer.save_model()


if __name__ == "__main__":
    main()
