#!/bin/sh
## =====you should change below before creating job======
region="" # project ID of Vertex AI
gcp_project="" # project ID of GCP

## =====you can change below before creating job======
git_hash=$(git rev-parse HEAD)
display_name=$git_hash # you can edit display name whatever you want. defaults is hash value of git
config_file="vertex_ai/configs/custom_job/dafault.yaml" # the path of config file


## =====build and push docker image======
# image_uri is obtained from config_file with the following code
pip install yaml
image_uri=$(python << EOF
import yaml

with open('${config_file}') as f:
    config = yaml.safe_load(f)
print(config['workerPoolSpecs']['containerSpec']['imageUri'])
EOF
)
echo "Use" $image_uri
docker build . -t $image_uri --platform=linux/x86_64
docker push $image_uri

## =====create custom job======
# https://cloud.google.com/sdk/gcloud/reference/ai/custom-jobs/create?hl=ja
gcloud ai custom-jobs create \
    --region=$region \
    --project=$gcp_project \
    --display-name=$display_name \
    --config=$config_file \
    --enable-web-access \
    --args=$@
