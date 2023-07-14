# Refer to documentation for more upload operations : https://docs.wandb.ai/guides/artifacts/construct-an-artifact

import wandb

run = wandb.init(entity="prabhupad26", project="sa-roberta", job_type="upload_dataset")
artifact = wandb.Artifact(name="sentiment-dataset", type="training_dataset")
artifact.add_file(
    local_path="/home/ppradhan/Documents/my_learnings/my_uni_stuffs/sa_data_storage/training.1600000.processed.noemoticon.csv"
)
artifact.add_file(
    local_path="/home/ppradhan/Documents/my_learnings/my_uni_stuffs/bdp2_apr22_exam-bdp2_apr22_group_2/sa_app/config_files/mapping.txt"
)
run.log_artifact(artifact)
