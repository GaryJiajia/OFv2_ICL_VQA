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

from PIL import ImageFilter
from open_flamingo.eval.coco_metric import compute_cider, postprocess_captioning_generation
from open_flamingo.eval.eval_datasets import VQADataset
from tqdm import tqdm

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


import re
def postprocess_new_vqa_generation(predictions):
    answer = re.split("Question|Answer|Short|Caption|Output|Declaration|Declarative|Sentence|Mask|Long|answer", predictions, 1)[0]
    answer = re.split(", ", answer, 1)[0]
    answer = re.split("\.", answer, 1)[0]
    # answer = re.split(" ", answer, 1)[0]
    return answer

def get_vqa_declaration_prompt(question, answer=None) -> str:
        return f"<image>Declarative sentence:{question} Short answer:{answer if answer is not None else ''}{'<|endofchunk|>' if answer is not None else ''}"

def get_vqa_and_declaration_prompt(question, declaration, answer=None) -> str:
        return f"<image>Question:{question} Declarative sentence:{declaration} Short answer:{answer if answer is not None else ''}{'<|endofchunk|>' if answer is not None else ''}"

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
    print(device_id)
    eval_model.set_device(device_id)
    eval_model.init_distributed()

    if args.model != "open_flamingo" and args.shots != [0]:
        raise ValueError("Only 0 shot eval is supported for non-open_flamingo models")

    if len(args.trial_seeds) != args.num_trials:
        raise ValueError("Number of trial seeds must be == number of trials.")

    results = defaultdict(list)
    datastes_name = (
    "vizwiz" if args.eval_vizwiz 
    else "ok_vqa" if args.eval_ok_vqa 
    else "vqav2" if args.eval_vqav2
    else None
    )
    if datastes_name == None:
        raise ValueError("Unsupported dataset")
    
    # results["prompt"].append(get_vqa_declaration_prompt("question","answer"))
    # results["prompt"].append(get_vqa_and_declaration_prompt("question","declaration","answer"))

    print(f"Evaluating on {datastes_name}...")
    for shot in args.shots:
        scores = []
        for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
            vqa_score = evaluate_vqa(
                args=args,
                eval_model=eval_model,
                num_shots=shot,
                seed=seed,
                dataset_name=datastes_name,
            )
            if args.rank == 0:
                print(f"Shots {shot} Trial {trial} {datastes_name} score: {vqa_score}")
                scores.append(vqa_score)

        if args.rank == 0:
            print(f"Shots {shot} Mean {datastes_name} score: {np.nanmean(scores)}")
            results[datastes_name].append(
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

def Get_DQQR_TAGs_id(test_question_id, num_samples):
    with open('sqqr-tags-n-v-index.json', 'r') as json_file:
        similar_questions_dict = json.load(json_file)
    similar_train_question_ids = similar_questions_dict.get(test_question_id, [])
    result = similar_train_question_ids[:num_samples]
    return result

# def sample_batch_demos_from_query_set(query_set, num_samples, batch_size):
def sample_batch_demos_from_query_set(query_set, num_samples, batch, retrieval_type, clip):
    if not clip:
        return [random.sample(list(query_set), num_samples) for _ in range(len(batch["image"]))]
    else:
        if retrieval_type == "mix_img_cap":
            return [[query_set.id2item(id) for id in batch["SI"][i][:(num_samples//2)] + batch["SQ"][i][:(num_samples//2)]] for i in range(len(batch["image"]))]
        elif retrieval_type == "DQQR-TAG":
            return [[query_set.id2item(id) for id in Get_DQQR_TAGs_id(str(batch["question_id"][i]), num_samples)] for i in range(len(batch["image"]))]
        else:
            return [[query_set.id2item(id) for id in batch[retrieval_type][i][:num_samples]] for i in range(len(batch["image"]))]



def compute_effective_num_shots(num_shots, model_type, number):
    if model_type == "open_flamingo":
        return num_shots if num_shots > 0 else number
    return num_shots


def custom_collate_fn(batch):
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = [item[key] for item in batch]
    return collated_batch

def get_incorrect_answer(answer_list,correct_answer):
    candidate_answers = [answer for answer in answer_list if answer != correct_answer]
    return random.choice(candidate_answers)

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
    elif dataset_name == "ok_vqa":
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
    # question-answer information
    question2answer_list_path = f"/open_flamingo/eval/data/{dataset_name}/{dataset_name}_question_to_answer.json"
    question_id2answer_list_path = f"open_flamingo/eval/data/{dataset_name}/{dataset_name}_question_id2answer.json"
    # question -> answer_list
    if os.path.exists(question2answer_list_path):
        with open(question2answer_list_path, "r") as file:
            question2answer_list = json.load(file)
    # question_id -> answer_list
    if os.path.exists(question_id2answer_list_path):
        with open(question_id2answer_list_path, "r") as file:
            question_id2answer_list = json.load(file)


    control_signals = {"clip": True,
                       "retrieval_name": args.retrieval_name,
                       "retrieval_type": "SI", 
                       "mismatch_type":"normal", # normal ; answer ; image ; question; question-answer
                       "specification": False,
                       "declaration": False,
                       "add_declaration": False,
                       "gauss": True,
                       "None_ICE":False,
                       "order": "order"} # order; reverse
    print("control signals of prompt: ", control_signals)    

    if control_signals["None_ICE"]:
        effective_num_shots = compute_effective_num_shots(num_shots, args.model, 0)
    else:
        effective_num_shots = compute_effective_num_shots(num_shots, args.model, 2)

    test_dataloader = prepare_eval_samples(
        test_dataset,
        args.num_samples if args.num_samples > 0 else len(test_dataset),
        args.batch_size,
        seed,
    )

    random_uuid = "{}_{}-shot".format(control_signals["retrieval_name"], num_shots)

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
                    if control_signals["mismatch_type"] != "normal":
                        for ics in in_context_samples:
                            random_item = random.choice(train_dataset)
                            if control_signals["mismatch_type"] == "image":
                                ics["image"] = random_item["image"]
                                ics["image_id"] = random_item["image_id"]
                            if control_signals["mismatch_type"] == "question":
                                ics["question_id"] = random_item["question_id"]
                                ics["question"] = random_item["question"]
                            if control_signals["mismatch_type"] == "question-answer":
                                ics["question_id"] = random_item["question_id"]
                                ics["question"] = random_item["question"]
                                ics["answers"] = random_item["answers"]
                            if control_signals["mismatch_type"] == "answer":
                                question_id = ics["question_id"]
                                answer = ics["answers"][0]
                                question = ics["question"]
                                qid = str(question_id)
                                answer_list_1 = list(set(question2answer_list[question]))
                                answer_list_2 = question_id2answer_list[qid]
                                if answer == 'yes':
                                    incorrect_answer = 'no'
                                if answer == 'no':
                                    incorrect_answer = 'yes'
                                elif len(answer_list_1) > 1:
                                    incorrect_answer = get_incorrect_answer(answer_list_1, answer)
                                else:
                                    incorrect_answer = get_incorrect_answer(answer_list_2, answer)
                                ics["answers"][0] = incorrect_answer
                    context_images = [x["image"] for x in in_context_samples]
                else:
                    context_images = []
                if control_signals["gauss"]:
                    batch_images.append(context_images + [batch["image"][i].filter(ImageFilter.GaussianBlur(radius=15))])
                else:
                    batch_images.append(context_images + [batch["image"][i]])

                if control_signals["declaration"]:
                    for j in range(len(in_context_samples)):
                        in_context_samples[j]["question"] = train_declaration_datasets.get(str(in_context_samples[j]["question_id"]))
                    batch["question"][i] = val_declaration_datasets.get(str(batch["question_id"][i]))
                    
                if control_signals["declaration"]:
                    context_text = "".join(
                        [
                            get_vqa_declaration_prompt(
                                question=x["question"], answer=x["answers"][0]
                            )
                            for x in in_context_samples
                        ]
                    )
                elif control_signals["add_declaration"]:
                    context_text = "".join(
                        [
                            get_vqa_and_declaration_prompt(
                                question=x["question"], declaration=train_declaration_datasets.get(str(x["question_id"])), answer=x["answers"][0]
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
                            for x in in_context_samples
                        ]
                    )
                
                if control_signals["specification"]:
                    context_text = "According to the previous question and answer pair, " \
                                   "answer the final question. " + context_text

                # Keep the text but remove the image tags for the zero-shot case
                if num_shots == 0:
                    context_text = context_text.replace("<image>", "")

                if control_signals["declaration"]:
                    batch_text.append(context_text + get_vqa_declaration_prompt(question=batch["question"][i]))
                elif control_signals["add_declaration"]:
                    batch_text.append(context_text + get_vqa_and_declaration_prompt(question=batch["question"][i], declaration=val_declaration_datasets.get(str(batch["question_id"][i]))))
                else:
                    batch_text.append(context_text + eval_model.get_vqa_prompt(question=batch["question"][i]))

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
                    raise exception


            process_function = (
                postprocess_ok_vqa_generation
                if dataset_name == "ok_vqa"
                else postprocess_vqa_generation
            )
            if control_signals["declaration"]:
                process_function = postprocess_new_vqa_generation


            new_predictions = map(process_function, outputs)

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

        # save the predictions to a temporary file
        # random_uuid = str(uuid.uuid4())
        with open(f"{dataset_name}results_{random_uuid}.json", "w") as f:
            f.write(json.dumps(all_predictions, indent=4))


    if test_annotations_json_path is not None:
        acc = compute_vqa_accuracy(
            f"{dataset_name}results_{random_uuid}.json",
            test_questions_json_path,
            test_annotations_json_path,
        )
        # delete the temporary file
        # os.remove(f"{dataset_name}results_{random_uuid}.json")

    else:
        print("No annotations provided, skipping accuracy computation.")
        print("Temporary file saved to:", f"{dataset_name}results_{random_uuid}.json")
        acc = None

    return acc


if __name__ == "__main__":
    main()
