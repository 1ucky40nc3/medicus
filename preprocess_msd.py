import os
import sys
import time
import shutil
import logging
import argparse

from batchgenerators.utilities.file_and_folder_operations import load_json

import nnunet
from nnunet.paths import (
    nnUNet_cropped_data
)
from nnunet.experiment_planning import nnUNet_convert_decathlon_task
from nnunet.experiment_planning.utils import (
    split_4d,
    crop
)
from nnunet.experiment_planning.DatasetAnalyzer import DatasetAnalyzer
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from nnunet.preprocessing.sanity_checks import verify_dataset_integrity
from nnunet.training.model_restore import recursive_find_python_class


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        help="Directory of the task as well as the raw dataset."
    )    
    parser.add_argument(
        "--msd_task",
        action="store_true",
        default=False,
        help=(
            "State if the task is from the Medical Segmentation Decathlon (MSD)"
            "or if the dataset follows the MSD's structure in great detail."
        )
    )
    parser.add_argument(
        "--task_id",
        default=None,
        type=str,
        help=(
            "The task id of the preprocessed dataset in the format '{id:03d}'."
            "In None is provided the task id will be parsed."
            "By default MSD task names follow this structure: 'Task{id:02d}_{name}'"
            "If no id is specified, the task will be parsed in the expected format."
        )
    )
    parser.add_argument(
        "--num_processes",
        default=8,
        type=int,
        help="Number of threads in preprocessing."
    )
    parser.add_argument(
        "--planner3d",
        default="ExperimentPlanner3D_v21",
        type=str,
        help="The name for the 3D experiment nnUNet experiment planner."
    )
    parser.add_argument(
        "--planner2d",
        default="ExperimentPlanner2D_v21",
        type=str,
        help="The name for the 2D experiment nnUNet experiment planner."
    )
    parser.add_argument(
        "--verify_dataset_integrity",
        action="store_true",
        default=False,
        help="State if dataset integrity shall be checked."
    )
    parser.add_argument(
        "--nnUNet_raw_data_base",
        default="raw_data_base",
        type=str,
        help=(
            "Path of the raw dataset directory."
            "Note: This variable will be set as an environment variable."
        )
    )
    parser.add_argument(
        "--nnUNet_preprocessed",
        default="preprocessed",
        type=str,
        help=(
            "The path to the preprocessed data."
            "Note: This variable will be set as an environment variable."
        )
    )
    parser.add_argument(
        "--nnUNet_RESULTS_FOLDER",
        default="result",
        type=str,
        help=(
            "The path to the results directory."
            "Note: This variable will be set as an environment variable."
        )
    )

    return parser.parse_args()


def set_environ(
    args: argparse.Namespace
) -> None:
    print("set environ")
    os.environ["nnUNet_raw_data_base"] = args.nnUNet_raw_data_base
    os.environ["nnUNet_preprocessed"] = args.nnUNet_preprocessed
    os.environ["RESULTS_FOLDER"] = args.nnUNet_RESULTS_FOLDER


def parse_task_id(
    args: argparse.Namespace        
) -> str:
    if not args.task_id:
        task = args.task.split("/")[-1]
        task_id = task.split("Task")[-1]
        task_id = task_id.split("_")[0]
        task_id = int(task_id)
        return task_id

    return int(args.task_id)


def main():
    args = arguments()
    if 'google.colab' in sys.modules:
        logging.info(
            "If you are using colab runtime: "
            "You are required to set the env variables manually "
            "and provide their corresponding flags!"
        )
    set_environ(args)

    if not args.msd_task:
        raise NotImplementedError()
    
    # Remove hidden files from the MSD task directory
    nnUNet_convert_decathlon_task.crawl_and_remove_hidden_from_decathlon(args.task)
    task_id = parse_task_id(args)
    split_4d(args.task, args.num_processes, task_id)

    task_name = convert_id_to_task_name(task_id)
    # Optionally verify the dataset integrity
    if args.verify_dataset_integrity:
        time.sleep(5)
        verify_dataset_integrity(
            os.path.join(
                args.nnUNet_raw_data_base, 
                "nnUNet_raw_data", 
                task_name
            )
        )
    
    crop(task_name, False, args.num_processes)

    # Load the planner classes
    search_in = os.path.join(
        nnunet.__path__[0], 
        "experiment_planning"
    )
    # Find the 3D experiment planner class 
    planner_3d = recursive_find_python_class(
        [search_in], 
        args.planner3d, 
        current_module="nnunet.experiment_planning"
    )
    if planner_3d is None:
        raise ValueError("The 3D experiment planner class couldn't be found.")
    # Find the 2D experiment planner class
    planner_2d = recursive_find_python_class(
        [search_in], 
        args.planner2d, 
        current_module="nnunet.experiment_planning"
    )
    if planner_2d is None:
        raise ValueError("The 2D experiment planner class couldn't be found.")

    # Init dir variables
    cropped_dir = os.path.join(nnUNet_cropped_data, task_name)
    prep_dir = os.path.join(args.nnUNet_preprocessed, task_name)

    # Init the dataset analyzer
    ds_json = load_json(os.path.join(cropped_dir, "dataset.json"))
    modalities = list(ds_json["modality"].values())
    collect_intensity = any("ct" in mod.lower() for mod in modalities)
    dataset_analyzer = DatasetAnalyzer(
        cropped_dir, 
        overwrite=False,
        num_processes=args.num_processes
    )
    dataset_analyzer.analyze_dataset(collect_intensity)

    # Prepare the dir with preprocessed data
    os.makedirs(prep_dir, exist_ok=True)
    shutil.copy(
        os.path.join(
            args.nnUNet_raw_data_base,
            "nnUNet_cropped_data",
            task_name,
            "dataset_properties.pkl"
        ), 
        prep_dir
    )
    shutil.copy(
        os.path.join(
            args.nnUNet_raw_data_base,
            "nnUNet_raw_data",
            task_name, 
            "dataset.json"
        ), 
        prep_dir
    )
    
    # Execute the planning
    threads = (args.num_processes)*2

    exp_planner = planner_3d(cropped_dir, prep_dir)
    exp_planner.plan_experiment()
    exp_planner.run_preprocessing(threads)

    exp_planner = planner_2d(cropped_dir, prep_dir)
    exp_planner.plan_experiment()
    exp_planner.run_preprocessing(threads)


if __name__ == "__main__":
    main()