def parse(config) -> dict:
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