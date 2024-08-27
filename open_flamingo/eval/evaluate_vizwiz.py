import argparse
import importlib
import json
import os
import random
import uuid
from collections import defaultdict
import copy

from einops import repeat
import more_itertools
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from open_flamingo.eval.coco_metric import compute_cider, postprocess_captioning_generation

from tqdm import tqdm

import sys


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
from open_flamingo.eval.eval_parser import parser

import sys
def main():
    args, leftovers = parser.parse_known_args()

    #print(sys.path)
    module = importlib.import_module(f"open_flamingo.eval.models.{args.model}")

    model_args = {
        leftovers[i].lstrip("-"): leftovers[i + 1] for i in range(0, len(leftovers), 2)
    }
    eval_model = module.EvalModel(model_args)

    # set up distributed evaluation
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    device_id = init_distributed_device(args)
    # device_id = "cuda:" + str(model_args["device"])
    print(device_id)
    # device_id = torch.device(device_id)
    eval_model.set_device(device_id)
    eval_model.init_distributed()

    if args.model != "open_flamingo" and args.shots != [0]:
        raise ValueError("Only 0 shot eval is supported for non-open_flamingo models")

    if len(args.trial_seeds) != args.num_trials:
        raise ValueError("Number of trial seeds must be == number of trials.")

    results = defaultdict(list)

    if args.eval_vizwiz:
        print("Evaluating on VizWiz...")
        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                vizwiz_score = evaluate_vqa(
                    args=args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    dataset_name="vizwiz",
                )
                if args.rank == 0:
                    print(f"Shots {shot} Trial {trial} VizWiz score: {vizwiz_score}")
                    scores.append(vizwiz_score)

            if args.rank == 0:
                print(f"Shots {shot} Mean VizWiz score: {np.nanmean(scores)}")
                results["vizwiz"].append(
                    {"shots": shot, "trials": scores, "mean": np.nanmean(scores)}
                )

    if args.eval_ok_vqa:
        print("Evaluating on OK-VQA...")
        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                ok_vqa_score = evaluate_vqa(
                    args=args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    dataset_name="ok_vqa",
                )
                if args.rank == 0:
                    print(f"Shots {shot} Trial {trial} OK-VQA score: {ok_vqa_score}")
                    scores.append(ok_vqa_score)

            if args.rank == 0:
                print(f"Shots {shot} Mean OK-VQA score: {np.nanmean(scores)}")
                results["ok_vqa"].append(
                    {"shots": shot, "trials": scores, "mean": np.nanmean(scores)}
                )

    if args.eval_vqav2:
        print("Evaluating on VQAv2...")
        for shot in args.shots:
            scores = []
            print("shot: ", shot)
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
            json.dump(results, f,indent=4)


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

def get_query_set(subset_train_dataset, query_set_size, seed):
    np.random.seed(seed)
    query_set = np.random.choice(len(subset_train_dataset), query_set_size, replace=False)
    return [subset_train_dataset[i] for i in query_set]


def prepare_sub_train_dataset(train_dataset, indices_list):
    sub_dataset = torch.utils.data.Subset(train_dataset, indices_list)
    return sub_dataset

def prepare_eval_samples(test_dataset, num_samples, batch_size, seed):
    if len(test_dataset)<num_samples:
        num_samples = len(test_dataset)
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

def get_incorrect_answer(answer_list,correct_answer):
    candidate_answers = [answer for answer in answer_list if answer != correct_answer]
    return random.choice(candidate_answers)


def sample_batch_demos_from_query_set(query_set, num_samples, batch, retrieval_type, clip):
    if not clip:
        output = []
        for _ in batch:
            o = []
            a = random.sample(range(len(query_set)), num_samples)
            for j, i in enumerate(a):
                x = copy.deepcopy(query_set[i])
                o.append(x)
            output.append(o)
        return output
    else:
        return [[query_set.id2item(id) for id in batch[retrieval_type][i][:num_samples]] for i in
                    range(len(batch["image"]))]

def compute_effective_num_shots(num_shots, model_type):
    if model_type == "open_flamingo":
        return num_shots if num_shots > 0 else 2
    return num_shots


def custom_collate_fn(batch):
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = [item[key] for item in batch]
    return collated_batch


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
    """
    Evaluate a model on VQA datasets. Currently supports VQA v2.0, OK-VQA, VizWiz and TextVQA.

    Args:
        args (argparse.Namespace): arguments
        eval_model (BaseEvalModel): model to evaluate
        seed (int, optional): random seed. Defaults to 42.
        max_generation_length (int, optional): max generation length. Defaults to 5.
        num_beams (int, optional): number of beams to use for beam search. Defaults to 3.
        length_penalty (float, optional): length penalty for beam search. Defaults to -2.0.
        num_shots (int, optional): number of shots to use. Defaults to 8.
        dataset_name (string): type of vqa dataset: currently supports vqav2, ok_vqa. Defaults to vqav2.
    Returns:
        float: accuracy score
    """
    if dataset_name == "ok_vqa":
        train_image_dir_path = args.ok_vqa_train_image_dir_path
        train_questions_json_path = args.ok_vqa_train_questions_json_path
        train_annotations_json_path = args.ok_vqa_train_annotations_json_path
        test_image_dir_path = args.ok_vqa_test_image_dir_path
        test_questions_json_path = args.ok_vqa_test_questions_json_path
        test_annotations_json_path = args.ok_vqa_test_annotations_json_path

    elif dataset_name == "vizwiz":
        train_image_dir_path = args.vizwiz_train_image_dir_path
        train_questions_json_path = args.vizwiz_train_questions_json_path
        train_annotations_json_path = args.vizwiz_train_annotations_json_path
        test_image_dir_path = args.vizwiz_test_image_dir_path
        test_questions_json_path = args.vizwiz_test_questions_json_path
        test_annotations_json_path = args.vizwiz_test_annotations_json_path

    elif dataset_name == "vqav2":
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
        dataset_name=dataset_name,
    )

    test_dataset = VQADataset(
        image_dir_path=test_image_dir_path,
        question_path=test_questions_json_path,
        annotations_path=test_annotations_json_path,
        is_train=False,
        dataset_name=dataset_name,
    )

    # declarative sentences
    train_declaration_path = "declaration/train2014_declarative.json"
    val_declaration_path = "declaration/val2014_declarative.json"
    if os.path.exists(train_declaration_path):
        with open(train_declaration_path, 'r') as file:
            train_declaration_datasets = json.load(file)
    else:
        print("Train_declaration file not exist")
    if os.path.exists(val_declaration_path):
        with open(val_declaration_path, 'r') as file:
            val_declaration_datasets = json.load(file)
    else:
        print("Val_declaration file not exist")

    effective_num_shots = compute_effective_num_shots(num_shots, args.model)

    test_dataloader = prepare_eval_samples(
        test_dataset,
        args.num_samples if args.num_samples > 0 else len(test_dataset),
        args.batch_size,
        seed,
    )

    control_signals = {"clip": False,
                       "retrieval_type": "SI", # SI;SQ;SQA;SQA_rs;SQA_SI;
                       "instruction_type": "instruction_false", # instruction_false; instruction_2; instruction_3;
                       "declaration": False,
                       "order": "order", # order; reverse
                       "SQAQAR_type":"normal", # normal;replaceI;replaceQ;
                       "mismatch_type": "none"}   # answer;image;none;qa

    print("control signals of prompt: ", control_signals)

    if control_signals["clip"]:
        random_uuid = "PREDICTION_FILE_{}_{}_{}_declare{}_{}_replace_{}_mismatch_{}".format(control_signals["retrieval_type"],
                                                  num_shots,
                                                  control_signals["instruction_type"],
                                                  control_signals["declaration"],
                                                  control_signals["order"],
                                                  control_signals["SQAQAR_type"],
                                                  control_signals["mismatch_type"])
    else:
        random_uuid = "PREDICTION_FILE_RS_{}_instruct_{}_declare_{}_{}_mismatch_{}".format(num_shots,
                                                  control_signals["instruction_type"],
                                                  control_signals["declaration"],
                                                  control_signals["order"],control_signals["mismatch_type"])

    predictions = []

    np.random.seed(
        seed + args.rank
    )  # make sure each worker has a different seed for the random context samples
    if not os.path.exists(f"{dataset_name}results_{random_uuid}.json"):
        for batch in tqdm(
                test_dataloader,
                desc=f"Running inference {dataset_name}",
                disable=args.rank != 0,
        ):
            batch_demo_samples = sample_batch_demos_from_query_set(
                train_dataset, effective_num_shots, batch,
                control_signals["retrieval_type"], control_signals["clip"])

            batch_images = []
            batch_text = []
            for i in range(len(batch["image"])):
                in_context_samples = batch_demo_samples[i]

                if control_signals["order"] == "reverse":
                    in_context_samples.reverse()

                if num_shots > 0:
                    if control_signals["retrieval_type"] == "SQA" and control_signals["SQAQAR_type"] == "replaceI":
                        for ics in in_context_samples:
                            random_item = random.choice(train_dataset)
                            ics["image"] = random_item["image"]
                            ics["image_id"] = random_item["image_id"]

                    context_images = [x["image"] for x in in_context_samples]
                else:
                    context_images = []
                batch_images.append(context_images + [batch["image"][i]])

                if control_signals["declaration"]:
                    for j in range(len(in_context_samples)):
                        in_context_samples[j]["question"] = train_declaration_datasets.get(
                            str(in_context_samples[j]["question_id"]))
                        batch["question"][i] = val_declaration_datasets.get(str(batch["question_id"][i]))


                if control_signals["retrieval_type"] == "SQA" and control_signals["SQAQAR_type"] == "replaceQ":
                    for ics in in_context_samples:
                        ics['question'] = ics["QA_question"]

                # mismatch-qa
                if control_signals["mismatch_type"] == "qa":
                    from random import sample
                    for ics in in_context_samples:
                        random_ics = random.choice(train_dataset)
                        ics["question_id"] = random_ics["question_id"]
                        ics["question"] = random_ics["question"]
                        ics["answers"] = random_ics["answers"]


                context_text = "".join(
                    [
                        eval_model.get_vqa_prompt(
                            question=x["question"], answer=x["answers"][0]
                        )
                        for x in in_context_samples
                    ]
                )

                if control_signals["instruction_type"] == "instruction_1":
                    context_text = "According to the previous question and answer pair, " \
                                   "answer the final question. " + context_text

                if control_signals["instruction_type"] == "instruction_2":
                    context_text = "According to the visual information in <image> token, " \
                                   "answer the following question. " + context_text

                if control_signals["instruction_type"] == "instruction_3":
                    context_text = "Answer the question. "  + context_text

                # Keep the text but remove the image tags for the zero-shot case
                if num_shots == 0:
                    context_text = context_text.replace("<image>", "")

                batch_text.append(
                    context_text + eval_model.get_vqa_prompt(question=batch["question"][i])
                    # context_text + eval_model.get_vqa_declaration_prompt(question=batch["question"][i])
                )

            try:
                outputs = eval_model.get_outputs(
                    batch_images=batch_images,
                    batch_text=batch_text,
                    min_generation_length=min_generation_length,
                    max_generation_length=max_generation_length,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                )
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    print("An exception occurred:", exception)
                    raise exception 

            process_function = (
                postprocess_ok_vqa_generation
                if dataset_name == "ok_vqa"
                else postprocess_vqa_generation
            )

            new_predictions = map(process_function, outputs)

            # for new_prediction, sample_id in zip(new_predictions, batch["question_id"]):
            #     predictions.append({"answer": new_prediction, "question_id": sample_id})
            new_predictions = list(new_predictions)
            for i in range(len(batch["image"])):
                predictions.append({"answer": new_predictions[i], "question_id": batch["question_id"][i],
                                    "prompt_text": batch_text[i],
                                    "prompt_images": [img['image_id'] for img in batch_demo_samples[i]],
                                    "prompt_question_id": [qui['question_id'] for qui in batch_demo_samples[i]],
                                    "question": batch["question"][i]
                                    })


        # all gather
        all_predictions = [None] * args.world_size
        torch.distributed.all_gather_object(all_predictions, predictions)  # list of lists
        if args.rank != 0:
            return

        all_predictions = [
            item for sublist in all_predictions for item in sublist
        ]  # flatten

        with open(f"{dataset_name}results_{random_uuid}.json", "w") as f:
            f.write(json.dumps(all_predictions, indent=4))

    if test_annotations_json_path is not None:
        acc = compute_vqa_accuracy(
            f"{dataset_name}results_{random_uuid}.json",
            test_questions_json_path,
            test_annotations_json_path,
        )
        # delete the temporary file
        #os.remove(f"{dataset_name}results_{random_uuid}.json")

    else:
        print("No annotations provided, skipping accuracy computation.")
        print("Temporary file saved to:", f"{dataset_name}results_{random_uuid}.json")
        acc = None

    return acc




if __name__ == "__main__":
    main()
