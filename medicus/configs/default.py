import argparse


def parse_config(config, **kwargs) -> dict:
    return {
        "model": {
            "name": config["model"].__name__,
            "config": config["model_config"]
        },
        "loss_fn": config["loss_fn"].__name__,
        "optimizer": {
            "name": config["optimizer"].__name__,
            "config": config["optimizer_config"]
        },
        "lr_scheduler": {
            "name": config["lr_scheduler"].__name__,
            "config": config["lr_scheduler_config"]
        },
        "num_epochs": config["num_epochs"],
        "log_dir": config["log_dir"],
        "save_dir": config["save_dir"],
        "log_every": config["log_every"],
        "eval_every": config["eval_every"],
        "save_every": config["save_every"],
        "methods": config["methods"],
        "project": config["project"],
        "notes": config["notes"],
        "tags": config["tags"]
    }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Name of the dataset."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        help="Dataset directory"
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size per device"
    )
    parser.add_argument(
        "--model",
        default="medicus/configs/model/unet.json",
        type=str,
        help="A file describing your model config"
    )
    parser.add_argument(
        "--optim",
        default="medicus/configs/optimizers/adam.json",
        type=str,
        help="A file describing your optimizer config"
    )
    parser.add_argument(
        "--sched",
        default="medicus/configs/schedulers/linear.json",
        type=str,
        help="A file describing your optimizer config"
    )
    parser.add_argument(
        "--loss_fn",
        default="bce_and_softdiceloss",
        type=str,
        help="The given name of a loss function"
    )
    parser.add_argument(
        "--log_every",
        default=50,
        type=int,
        help="Log interval"
    )
    parser.add_argument(
        "--num_epochs",
        default=40,
        type=int,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--logging_method",
        default="tensorboard",
        nargs="+",
        type=str,
        help="A logging method (`tensorboard` or `wandb`)"
    )
    parser.add_argument(
        "--eval_every",
        default=500,
        type=int,
        help="The evaluation interval"
    )
    parser.add_argument(
        "--save_every",
        default=500,
        type=int,
        help="The for checkpoints of the model weights"
    )
    parser.add_argument(
        "--log_dir",
        default="runs/{}/logs",
        type=str,
        help="Directory for logging."
    )
    parser.add_argument(
        "--save_dir",
        default="runs/{}/checkpoints",
        type=str,
        help="Directory to save checkpoints in."
    )
    parser.add_argument(
        "--resume_from",
        default=None,
        type=str,
        help="Directory with checkpoint to resume from."
    )
    return parser

