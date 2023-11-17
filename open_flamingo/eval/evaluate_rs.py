import argparse
import importlib
import json
import os
import random
import uuid
from collections import defaultdict

from einops import repeat
import more_itertools
import numpy as np
import torch
# from sklearn.metrics import roc_auc_score

from open_flamingo.eval.coco_metric import compute_cider, postprocess_captioning_generation
from open_flamingo.eval.eval_datasets import (
    VQADataset,
)
from tqdm import tqdm


from open_flamingo.eval.eval_datasets import VQADataset
from open_flamingo.eval.classification_utils import (
    IMAGENET_CLASSNAMES,
    IMAGENET_1K_CLASS_ID_TO_LABEL,
    HM_CLASSNAMES,
    HM_CLASS_ID_TO_LABEL,
)

from open_flamingo.eval.eval_model import BaseEvalModel

from open_flamingo.eval.ok_vqa_utils import postprocess_ok_vqa_generation
from open_flamingo.src.flamingo import Flamingo
from open_flamingo.eval.vqa_metric import compute_vqa_accuracy, postprocess_vqa_generation

from open_flamingo.train.distributed import init_distributed_device, world_info_from_env

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    type=str,
    help="Model name. Currently only `OpenFlamingo` is supported.",
    default="open_flamingo",
)
parser.add_argument(
    "--results_file", type=str, default=None, help="JSON file to save results"
)
# Trial arguments
parser.add_argument("--shots", nargs="+", default=[0, 4, 8, 16, 32], type=int)
parser.add_argument(
    "--num_trials",
    type=int,
    default=1,
    help="Number of trials to run for each shot using different demonstrations",
)
parser.add_argument(
    "--trial_seeds",
    nargs="+",
    type=int,
    default=[42],
    help="Seeds to use for each trial for picking demonstrations and eval sets",
)
parser.add_argument(
    "--num_samples",
    type=int,
    default=-1,
    help="Number of samples to evaluate on. -1 for all samples.",
)
parser.add_argument(
    "--query_set_size", type=int, default=2048, help="Size of demonstration query set"
)

parser.add_argument("--batch_size", type=int, default=8)

parser.add_argument(
    "--no_caching_for_classification",
    action="store_true",
    help="Use key-value caching for classification evals to speed it up. Currently this doesn't underperforms for MPT models.",
)

# Per-dataset evaluation flags
parser.add_argument(
    "--eval_vqav2",
    action="store_true",
    default=True,
    help="Whether to evaluate on VQAV2.",
)
# Dataset arguments
## VQAV2 Dataset
parser.add_argument(
    "--vqav2_train_image_dir_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqav2_train_questions_json_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqav2_train_annotations_json_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqav2_test_image_dir_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqav2_test_questions_json_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqav2_test_annotations_json_path",
    type=str,
    default=None,
)
# Distributed evaluation
parser.add_argument(
    "--dist-url",
    default="env://",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument(
    "--horovod",
    default=False,
    action="store_true",
    help="Use horovod for distributed training.",
)
parser.add_argument(
    "--no-set-device-rank",
    default=False,
    action="store_true",
    help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
)
parser.add_argument(
    "--device",
    type=int,
    default=0,
    help="device of GPUs.",
)
parser.add_argument(
    "--retrieval_name",
    type=str,
    default="name",
    help="name of json result file",
)
parser.add_argument(
    "--dataset_name",
    type=str,
    default="vqav2",
    help="name of dataset name"
)


def main():
    args, leftovers = parser.parse_known_args()
    module = importlib.import_module(f"open_flamingo.eval.models.{args.model}")

    model_args = {
        leftovers[i].lstrip("-"): leftovers[i + 1] for i in range(0, len(leftovers), 2)
    }
    eval_model = module.EvalModel(model_args)

    # set up distributed evaluation
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    device_id = init_distributed_device(args)
    # device_id = "cuda:" + str(args.device_number)
    print(device_id)
    eval_model.set_device(device_id)
    eval_model.init_distributed()

    if args.model != "open_flamingo" and args.shots != [0]:
        raise ValueError("Only 0 shot eval is supported for non-open_flamingo models")

    if len(args.trial_seeds) != args.num_trials:
        raise ValueError("Number of trial seeds must be == number of trials.")

    results = defaultdict(list)

    if args.eval_vqav2:
        print("Evaluating on VQAv2...")
        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                vqa_score = evaluate_vqa(
                    args=args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    dataset_name="vqav2",
                )
                if args.rank == 0:
                    print(f"Shots {shot} Trial {trial} VQA score: {vqa_score}")
                    scores.append(vqa_score)

            if args.rank == 0:
                print(f"Shots {shot} Mean VQA score: {np.nanmean(scores)}")
                results["vqav2"].append(
                    {"shots": shot, "trials": scores, "mean": np.nanmean(scores)}
                )

    if args.rank == 0 and args.results_file is not None:
        with open(args.results_file, "w") as f:
            json.dump(results, f)


def get_random_indices(num_samples, query_set_size, full_dataset, seed):
    if num_samples + query_set_size > len(full_dataset):
        raise ValueError(
            f"num_samples + query_set_size must be less than {len(full_dataset)}"
        )

    # get a random subset of the dataset
    np.random.seed(seed)
    random_indices = np.random.choice(
        len(full_dataset), num_samples + query_set_size, replace=False
    )
    return random_indices


def get_query_set(train_dataset, query_set_size, seed):
    np.random.seed(seed)
    query_set = np.random.choice(len(train_dataset), query_set_size, replace=False)
    return [train_dataset[i] for i in query_set]


def prepare_eval_samples(test_dataset, num_samples, batch_size, seed):
    np.random.seed(seed)
    random_indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    dataset = torch.utils.data.Subset(test_dataset, random_indices)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=custom_collate_fn,
    )
    return loader


def sample_batch_demos_from_query_set(query_set, num_samples, batch_size):
    return [random.sample(query_set, num_samples) for _ in range(batch_size)]


def compute_effective_num_shots(num_shots, model_type):
    if model_type == "open_flamingo":
        return num_shots if num_shots > 0 else 2
    return num_shots


def custom_collate_fn(batch):
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = [item[key] for item in batch]
    return collated_batch

import re
def postprocess_vqa_new_generation(predictions):
    answer = re.split("Question|Answer|Short|Caption|Output|mask", predictions, 1)[0]
    answer = re.split(", ", answer, 1)[0]
    # answer = re.split("\.", answer, 1)[0]
    return answer

def get_vqa_declaration_prompt(question, answer=None) -> str:
        return f"<image>{question} mask:{answer if answer is not None else ''}{'<|endofchunk|>' if answer is not None else ''}"

def evaluate_vqa(
    args: argparse.Namespace,
    eval_model: BaseEvalModel,
    seed: int = 42,
    min_generation_length: int = 0,
    max_generation_length: int = 5,
    num_beams: int = 3,
    length_penalty: float = 0.0,
    num_shots: int = 8,
    dataset_name: str = "vqav2",
):
    if dataset_name == "vqav2":
        train_image_dir_path = args.vqav2_train_image_dir_path
        train_questions_json_path = args.vqav2_train_questions_json_path
        train_annotations_json_path = args.vqav2_train_annotations_json_path
        test_image_dir_path = args.vqav2_test_image_dir_path
        test_questions_json_path = args.vqav2_test_questions_json_path
        test_annotations_json_path = args.vqav2_test_annotations_json_path
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_dataset = VQADataset(
        image_dir_path=train_image_dir_path,
        question_path=train_questions_json_path,
        annotations_path=train_annotations_json_path,
        is_train=True,
        dataset_name=args.dataset_name,
    )

    test_dataset = VQADataset(
        image_dir_path=test_image_dir_path,
        question_path=test_questions_json_path,
        annotations_path=test_annotations_json_path,
        is_train=False,
        dataset_name=args.dataset_name,
    )

    control_signals = {"retrieval_name": args.retrieval_name,
                       "declaration": False,
                       "instruction": True}
    random_uuid = "{}_{}-shot".format(control_signals["retrieval_name"], num_shots)
    print("control signals of prompt: ", control_signals)

    # declarative sentences
    train_declaration_path = "declaration/train2014_declarative.json"
    val_declaration_path = "declaration/val2014_declarative.json"
    if os.path.exists(train_declaration_path):
        with open(train_declaration_path, 'r') as file:
            train_declaration_datasets = json.load(file)
    else:print("Train_declaration file not exist")
    if os.path.exists(val_declaration_path):
        with open(val_declaration_path, 'r') as file:
            val_declaration_datasets = json.load(file)
    else:print("Val_declaration file not exist")

    effective_num_shots = compute_effective_num_shots(num_shots, args.model)

    test_dataloader = prepare_eval_samples(
        test_dataset,
        args.num_samples if args.num_samples > 0 else len(test_dataset),
        args.batch_size,
        seed,
    )

    in_context_samples = get_query_set(train_dataset, args.query_set_size, seed)
    predictions = []

    np.random.seed(
        seed + args.rank
    )  # make sure each worker has a different seed for the random context samples
    for batch in tqdm(
        test_dataloader,
        desc=f"Running inference {dataset_name}",
        disable=args.rank != 0,
    ):
        batch_demo_samples = sample_batch_demos_from_query_set(
            in_context_samples, effective_num_shots, len(batch["image"])
        )

        batch_images = []
        batch_text = []
        for i in range(len(batch["image"])):
            if num_shots > 0:
                context_images = [x["image"] for x in batch_demo_samples[i]]
            else:
                context_images = []
            batch_images.append(context_images + [batch["image"][i]])

            if control_signals["declaration"]:
                for j in range(len(batch_demo_samples[i])):
                    batch_demo_samples[i][j]["question"] = train_declaration_datasets.get(str(batch_demo_samples[i][j]["question_id"]))
                batch["question"][i] = val_declaration_datasets.get(str(batch["question_id"][i]))

            
            if control_signals["declaration"]:
                context_text = "".join(
                    [
                        get_vqa_declaration_prompt(
                            question=x["question"], answer=x["answers"][0]
                        )
                        for x in batch_demo_samples[i]
                    ]
                )
            else:
                context_text = "".join(
                    [
                        eval_model.get_vqa_prompt(
                            question=x["question"], answer=x["answers"][0]
                        )
                        for x in batch_demo_samples[i]
                    ]
                )
            if control_signals["instruction"]:
                context_text = "For the upcoming tasks, you'll be provided image-text pairs. Digest these pairs carefully. Later, an image along with a question will be presented. Combine your understanding from the pairs and the new image to answer." + context_text
            # Keep the text but remove the image tags for the zero-shot case
            if num_shots == 0:
                context_text = context_text.replace("<image>", "")

            if control_signals["declaration"]:
                batch_text.append(context_text + get_vqa_declaration_prompt(question=batch["question"][i]))
            else:
                batch_text.append(context_text + eval_model.get_vqa_prompt(question=batch["question"][i]))

        outputs = eval_model.get_outputs(
            batch_images=batch_images,
            batch_text=batch_text,
            min_generation_length=min_generation_length,
            max_generation_length=max_generation_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
        )

        process_function = (
            postprocess_ok_vqa_generation
            if args.dataset_name == "ok_vqa"
            # else postprocess_vqa_new_generation
            else postprocess_vqa_generation
        )

        # if control_signals["declaration"]:
        # process_function = postprocess_vqa_new_generation

        # process_function = postprocess_ok_vqa_generation

        new_predictions = map(process_function, outputs)

        for new_prediction, sample_id, batch_text, batch_demo_samples, question in zip(new_predictions, batch["question_id"], batch_text, batch_demo_samples, batch["question"]):
            predictions.append({"answer": new_prediction, "question_id": sample_id,
                                "prompt_text": batch_text,
                                "prompt_images": [img['image_id'] for img in batch_demo_samples],
                                "question": question
                                })

    # all gather
    all_predictions = [None] * args.world_size
    torch.distributed.all_gather_object(all_predictions, predictions)  # list of lists
    if args.rank != 0:
        return

    all_predictions = [
        item for sublist in all_predictions for item in sublist
    ]  # flatten

    # save the predictions to a temporary file
    # random_uuid = str(uuid.uuid4())
    with open(f"{dataset_name}_{random_uuid}.json", "w") as f:
        f.write(json.dumps(all_predictions, indent=4))

    if test_annotations_json_path is not None:
        acc = compute_vqa_accuracy(
            f"{dataset_name}_{random_uuid}.json",
            test_questions_json_path,
            test_annotations_json_path,
        )
        # delete the temporary file
        # os.remove(f"{dataset_name}_{random_uuid}.json")

    else:
        print("No annotations provided, skipping accuracy computation.")
        print("Temporary file saved to:", f"{dataset_name}_{random_uuid}.json")
        acc = None

    return acc

if __name__ == "__main__":
    main()
