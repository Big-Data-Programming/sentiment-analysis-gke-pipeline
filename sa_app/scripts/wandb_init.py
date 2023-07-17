# Refer to documentation for more upload operations : https://docs.wandb.ai/guides/artifacts/construct-an-artifact
import argparse
from typing import List

import wandb


def upload_artifact(entity, project, artifact_name, artifact_type, artifact_locations: List[str]):
    run = wandb.init(entity=entity, project=project, job_type="upload_dataset")
    artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
    for artifact_location in artifact_locations:
        artifact.add_file(local_path=artifact_location)
    run.log_artifact(artifact)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--entity",
        default="prabhupad26",
        type=str,
        help="username",
    )
    parser.add_argument(
        "--project",
        default="sa-roberta",
        type=str,
        help="username",
    )
    parser.add_argument(
        "--artifact_name",
        default="sentiment-dataset",
        type=str,
        help="username",
    )
    parser.add_argument(
        "--artifact_type",
        default="training_dataset",
        type=str,
        help="username",
    )
    parser.add_argument(
        "--artifact_locations",
        default=[
            "/home/ppradhan/Documents/my_learnings/my_uni_stuffs/sa_data_storage/training.1600000.processed.noemoticon.csv",
            "/home/ppradhan/Documents/my_learnings/my_uni_stuffs/bdp2_apr22_exam-bdp2_apr22_group_2/sa_app/config_files/mapping.txt",
        ],
        type=str,
        nargs="+",
        help="username",
    )

    return parser.parse_args()


if __name__ == "__main__":
    wandb.login()
    args = parse_args()
    print("Uploading dataset")
    upload_artifact(entity=args.entity, project=args.project, artifact_locations=args.artifact_locations)
