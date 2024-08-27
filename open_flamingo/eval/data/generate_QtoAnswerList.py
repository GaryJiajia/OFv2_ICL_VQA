import numpy as np
import json
from collections import defaultdict
from open_flamingo.eval.eval_datasets import VQADataset

train_image_dir_path = path_for_train_image_dir_path
train_questions_json_path = path_for_train_questions_json_path
train_annotations_json_path = path_for_train_annotations_json_path
test_image_dir_path = path_for_test_image_dir_path
test_questions_json_path = path_for_test_questions_json_path
test_annotations_json_path = path_for_test_annotations_json_path
dataset_name = "vqav2"
full_dataset = VQADataset(
    image_dir_path=train_image_dir_path,
    question_path=train_questions_json_path,
    annotations_path=train_annotations_json_path,
    is_train=True,
    dataset_name=dataset_name,
    image_dir_path2=test_image_dir_path
    )

question2answer_list = defaultdict(dict)

count = 0
for idx, item in enumerate(full_dataset):
    question = item["question"]
    answer = item["answers"][0]
    if question not in question2answer_list:
        question2answer_list[question] = []
    question2answer_list[question].append(answer)
    count +=1


output_file_path = "vqav2_question_to_answer.json"

with open(output_file_path, "w") as json_file:
    json.dump(question2answer_list, json_file,indent=4)

print(f"Data written to {output_file_path}")


