import json
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import numpy as np

class VQADataset(Dataset):
    def __init__(
        self, image_dir_path, question_path, annotations_path, is_train, dataset_name
    ):
        self.questions = json.load(open(question_path, "r"))["questions"]
        if annotations_path is not None:
            self.answers = json.load(open(annotations_path, "r"))["annotations"]
        else:
            self.answers = None
        self.image_dir_path = image_dir_path
        self.is_train = is_train
        self.dataset_name = dataset_name
        if self.dataset_name in {"vqav2", "ok_vqa"}:
            self.img_coco_split = self.image_dir_path.strip("/").split("/")[-1]
            assert self.img_coco_split in {"train2014", "val2014", "test2015"}
        if self.dataset_name in {"vqav2"} and self.is_train == False:
            retrieval_path = "retrieval_results/vqav2/validation_sim_32.npy"
            self.retrieval_set = np.load(retrieval_path, allow_pickle=True).item()
        if self.dataset_name in {"ok_vqa"} and self.is_train == False:
            retrieval_path = "retrieval_results/okvqa_validation_SQQR.npy"
            self.retrieval_set = np.load(retrieval_path, allow_pickle=True).item()
        if self.dataset_name in {"vizwiz"} and self.is_train == False:
            retrieval_path = "retrieval_results/vizwiz_validation_SQQR.npy"
            self.retrieval_set = np.load(retrieval_path, allow_pickle=True).item()


    def __len__(self):
        return len(self.questions)

    def id2item(self, idx):
        question = self.questions[idx]
        answers = self.answers[idx]
        img_path = self.get_img_path(question)
        image = Image.open(img_path)
        return {
            "image": image,
            "image_id": question['image_id'],
            "question": question["question"],
            "answers": [a["answer"] for a in answers["answers"]],
            "question_id": question["question_id"],
        }

    def get_img_path(self, question):
        if self.dataset_name in {"vqav2", "ok_vqa"}:
            return os.path.join(
                self.image_dir_path,
                f"COCO_{self.img_coco_split}_{question['image_id']:012d}.jpg"
                if self.is_train
                else f"COCO_{self.img_coco_split}_{question['image_id']:012d}.jpg",
            )
        elif self.dataset_name == "vizwiz":
            return os.path.join(self.image_dir_path, question["image_id"])
        elif self.dataset_name == "textvqa":
            return os.path.join(self.image_dir_path, f"{question['image_id']}.jpg")
        else:
            raise Exception(f"Unknown VQA dataset {self.dataset_name}")

    def __getitem__(self, idx):
        question = self.questions[idx]
        img_path = self.get_img_path(question)
        image = Image.open(img_path)
        image.load()
        results = {
            "image": image,
            "image_id": question['image_id'],
            "question": question["question"],
            "question_id": question["question_id"],
        }
        if self.answers is not None:
            answers = self.answers[idx]
            results["answers"] = [a["answer"] for a in answers["answers"]]
        # if you need other retrieval method, add it into the results.
        if self.dataset_name in {"vqav2"} and self.is_train == False:
            question_id = question['question_id']
            results["SI"] = [i[2] for i in self.retrieval_set[question_id]["SI"]]
            results["SI_Q"] = [i[2] for i in self.retrieval_set[question_id]["SI_Q"]]
            results["SQ"] = [i[2] for i in self.retrieval_set[question_id]["SQ"]]
            results["SQ_I"] = [i[2] for i in self.retrieval_set[question_id]["SQ_I"]]
            results["SI_1"] = [i[2] for i in self.retrieval_set[question_id]["SI_1"]]
            results["SI_2"] = [i[2] for i in self.retrieval_set[question_id]["SI_2"]]
        if self.dataset_name in {"ok_vqa"} and self.is_train == False:
            question_id = question['question_id']
            results["SQ"] = [i[2] for i in self.retrieval_set[question_id]["SQ"]]
            results["SQ_I"] = [i[2] for i in self.retrieval_set[question_id]["SQ_I"]]
        if self.dataset_name in {"vizwiz"} and self.is_train == False:
            question_id = question['question_id']
            results["SQ"] = [i[2] for i in self.retrieval_set[question_id]["SQ"]]
            results["SQ_I"] = [i[2] for i in self.retrieval_set[question_id]["caption_image"]]
        return results
