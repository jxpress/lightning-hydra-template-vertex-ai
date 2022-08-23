#!/bin/sh

## =====you should change below before creating job======
region="" # project ID of Vertex AI
gcp_project="" # project ID of GCP


## =====you can change below before creating job======
config_file="vertex_ai/configs/hparams_tuning/default.yaml" # the path of config file
git_hash=$(git rev-parse HEAD)
display_name=$git_hash # you can edit display name whatever you want. defaults is hash value of git

## =====get some parameters from config file======
# I don't know why but somehow maxTrialCount and parallelTrialCount in config don't recognized by gcloud
# so that these parameters are obtained from config_file

maxTrialCount=$(python << EOF
import yaml

with open('${config_file}') as f:
    config = yaml.safe_load(f)
print(config['maxTrialCount'])
EOF
)

parallelTrialCount=$(python << EOF
import yaml

with open('${config_file}') as f:
    config = yaml.safe_load(f)
print(config['parallelTrialCount'])
EOF
)

## =====build and push docker image======
# image_uri is obtained from config_file with the following code
pip install yaml
image_uri=$(python << EOF
import yaml

with open('${config_file}') as f:
    config = yaml.safe_load(f)
print(config['trialJobSpec']['workerPoolSpecs']['containerSpec']['imageUri'])
EOF
)
echo "USE ${image_uri}"

docker build . -t $image_uri  --platform=linux/x86_64 
docker push $image_uri

# https://cloud.google.com/sdk/gcloud/reference/ai/hp-tuning-jobs/create?hl=ja
gcloud ai hp-tuning-jobs create \
    --display-name=$display_name \
    --region=$region \
    --project=$gcp_project \
    --config=$config_file \
    --max-trial-count=$maxTrialCount \
    --parallel-trial-count=$parallelTrialCount \
    --enable-web-access
