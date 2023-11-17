import argparse
import importlib
import json
import os

from open_flamingo.eval.vqa_metric_new import compute_vqa_accuracy

from open_flamingo.train.distributed import world_info_from_env

parser = argparse.ArgumentParser()

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

## VQAV2 Dataset
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

parser.add_argument(
    "--results_file_name",
    type=str,
    default="name",
    help="name of json result file",
)
parser.add_argument(
    "--metric_type",
    type=int,
    default=10,
    help="metric_type",
)

def main():
    args, leftovers = parser.parse_known_args()

    args.local_rank, args.rank, args.world_size = world_info_from_env()

    if len(args.trial_seeds) != args.num_trials:
        raise ValueError("Number of trial seeds must be == number of trials.")

    # results = defaultdict(list)
    if args.eval_vqav2:
        if args.metric_type != 10:
            for shot in args.shots:
                scores = []
                filename = args.results_file_name.replace("[number]", str(shot))
                vqa_score = evaluate_vqa(
                    args=args,
                    filename = filename, 
                    metric_type = args.metric_type
                )
                if args.rank == 0:
                    print(f"Shots {shot} VQA score: {vqa_score}")
                    scores.append(vqa_score)
        else:
            print("Eval type: Old")
            for shot in args.shots:
                scores = []
                filename = args.results_file_name.replace("[number]", str(shot))
                vqa_score = evaluate_vqa(
                    args=args,
                    filename = filename, 
                    metric_type = 0
                )
                if args.rank == 0:
                    print(f"Shots {shot} VQA score: {vqa_score}")
                    scores.append(vqa_score)
            print("Eval type: New")
            for shot in args.shots:
                scores = []
                filename = args.results_file_name.replace("[number]", str(shot))
                vqa_score = evaluate_vqa(
                    args=args,
                    filename = filename, 
                    metric_type = 1
                )
                if args.rank == 0:
                    print(f"Shots {shot} VQA score: {vqa_score}")
                    scores.append(vqa_score)


def evaluate_vqa(
    args: argparse.Namespace,
    filename,
    metric_type
):
    test_questions_json_path = args.vqav2_test_questions_json_path
    test_annotations_json_path = args.vqav2_test_annotations_json_path

    if test_annotations_json_path is not None:
        acc = compute_vqa_accuracy(
            filename,
            test_questions_json_path,
            test_annotations_json_path,
            metric_type
        )
    return acc


if __name__ == "__main__":
    main()
