import argparse


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--dataset_name",
        default="shapes",
        type=str,
        help="Name of the dataset."
    )
    parser.add_argument(
        "--dataset_dir",
        default="datasets/shapes",
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
        "--data",
        default="medicus/configs/data/numpy.json",
        type=str,
        help="A file describing your data config"
    )
    parser.add_argument(
        "--model",
        default="medicus/configs/models/unet.json",
        type=str,
        help="A file describing your model config"
    )
    parser.add_argument(
        "--loss",
        default="medicus/configs/objectives/unet.json",
        type=str,
        help="The given name of a loss function config"
    )
    parser.add_argument(
        "--optim",
        default="medicus/configs/optimizers/adam.json",
        type=str,
        help="A file describing your optimizer config"
    )
    parser.add_argument(
        "--sched",
        default="medicus/configs/schedules/steplr.json",
        type=str,
        help="A file describing your optimizer config"
    )
    parser.add_argument(
        "--config",
        nargs="*",
        default="",
        type=str,
        help="Additional config outside of config files."
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
        default=["tensorboard"],
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
    parser.add_argument(
        "--project",
        default="medicus",
        type=str,
        help="Name of the project for Weights & Biases logging."
    )
    return parser

