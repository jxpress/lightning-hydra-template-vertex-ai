# Lightning Hydra Template Vertex AI

[PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) and  [Hydra](https://github.com/facebookresearch/hydra), which is a learning framework and hyperparameter management package, can provide various benefits, such as parallel learning with only a few lines of code changes. The excellent [train template code](https://github.com/ashleve/lightning-hydra-template) created based on these two packages is available for public use. For more information on PyTorch Lightning and Hydra, see [README](documents/README.original.md) of the template code.

This repository offer the code that makes the template run on [Vertex AI](https://cloud.google.com/vertex-ai).

the Japanese version of the README is [here](/documents/README_ja.md)

![main_theme](/documents/images/main_readme.png)
<br>

# 💡　Reason for opening the repository to the public


Vertex AI is an integrated machine learning platform on Google Cloud Platform, and by using Vertex AI, the following can be easily executed. ( (★) can be implemented in this repository, and we will also open sample code for (*))
- Training that the GPU is activated only during training (★)
- Parallel training for hyperparameter tuning (★)
- Separate process such as data preprocessing, training, evaluation, and deployment, and connect each of them in a pipeline for learning (*)
- Periodic or triggered training on GPU machines


However, Vertex AI and Hydra are incompatible because of the different way of passing command line arguments.
In order to run code written in Hydra on Vertex AI, we need to devise a way to run it.
In this repository, we have provided the code that has been devised so that you can learn with Vertex AI without difficulty.

For more information on the problem and solution, please see [this blog](/documents/translated_blog.md).
<br>

# 🚀  How to use this repository
### step 1. Edit the template code and create your own train code (Optional).
If we want to create our own AI, you have to edit the template code and make sure the training is complete.

If you just want to check how it works with Vertex AI, you can run the template code without editing, and it will train the MNIST classification.

<br>

### step 2. confirm that training is excecutable with Docker Image.
Vertex AI uses Docker Image for training, so it is necessary to confirm the training on Docker Image.
At that time, you can confirm that by typing below in root directory.
```bash
make train-in-docker
```
Option such as checking operation on GPU can be adjusted in [docker-compose.yaml](/docker-compose.yaml).

<br>


### step 3. Prepare a GCP account.
If you do not have a GCP account, please prepare a GCP account from [here](https://cloud.google.com/docs/get-started).
This repository uses [Vertex AI](https://cloud.google.com/vertex-ai/docs/start) and [Artifact Registry](https://cloud.google.com/artifact-registry). Please activate the respective APIs in GCP.

Next, [create a docker repository](https://cloud.google.com/artifact-registry/docs/repositories/create-repos#overview)) to push Docker Images to the Artifact Registry.

Then [determine the name of the Image](https://cloud.google.com/artifact-registry/docs/docker/pushing-and-pulling).

<br>

### step 4-1. Run a custom job

- Set the name and tag of the Image determined in step 3 in imageUri of [vertex_ai/configs/custom_job/dafault.yaml](/vertex_ai/configs/custom_job/dafault.yaml).
- Set region, gcp_project in [vertex_ai/scripts/custom_job/create_job.sh](/vertex_ai/scripts/custom_job/create_job.sh).
- In the root folder, type
```bash
make create-custom-job
```
in the root folder.
Docker build and push will be performed, and the custom job of Vertex AI will be started with the pushed image.
You can check the training status at [CUSTOM JOBS](https://console.cloud.google.com/vertex-ai/training/custom-jobs) in the Vertex AI training section of GCP.

<br>


### step 4-2. Run a hyperparameter tuning job

- Set the name and tag of the Image determined in step 3 in imageUri of  [vertex_ai/configs/hparams_tuning/default.yaml](/vertex_ai/configs/hparams_tuning/default.yaml).
- Set the metrics that you want optimize in [configs/hparams_search/vertex_ai.yaml](/configs/hparams_search/vertex_ai.yaml).
- Set region, gcp_project in [vertex_ai/scripts/hparams_tuning/create_job.sh](/vertex_ai/scripts/hparams_tuning/create_job.sh)
- In the root folder, type
```bash
make create-hparams-tuning-job
```
in the root folder.
Docker build and push will be performed, and the hyperparameter tuning job of Vertex AI will be started with the pushed image.

You can check the training status at [HYPERPARAMETER TUNING JOBS](https://console.cloud.google.com/vertex-ai/training/hyperparameter-tuning-jobs) in the Vertex AI training section of GCP.

<br>



# 🔧　Changes
The following changes have been made in this repository from [train template code](https://github.com/ashleve/lightning-hydra-template).
- configs/hparams_search/vertex_ai.yaml
    - Used in hyperparameter tuning of Vertex AI
- Makefile 
    - Add code related to docker and Vertex AI
- folder and code for Vertex AI
    - configs
        - Add yaml file related to settings.
    - script
        - Add code to excecute train job in Vertex AI
- requirements.txt
    - Add package for Vertex AI
- README.md
    - Add README.md. Original README is moved to Documents folder
- Documents
    - Move the original README.md
    - Add the Japanese version of README.md
    - translated blog
        - English translation of a detailed blog about Hydra and Vertex AI.


# 📝 Appendix


JX PRESS Corporation has created and use the training template code in order to enhance team development capability and development speed.

We have created this repository by transferring only the code for training with Vertex AI from JX's training template code to [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template).

For more information on JX's training template code, see [How we at JX PRESS Corporation devise for team development of R&D that tends to become a genus](https://tech.jxpress.net/entry/2021/10/27/160154) and [PyTorch Lightning explained by a heavy user](https://techjxpress.net/entry/2021/11/17/112214). (Now these blogs are written in Japanese. If you want to see, please translate it into your language. We would like to translate it in English and publish it someday)
<br>

# 😍 Main contributors
The transfer to this repository was done by [Yongtae](https://github.com/Yongtae723), but the development was conceived and proposed by [Yongtae](https://github.com/Yongtae723) and [near129](https://github.com/near129) led the code development.

<br>

### 🔍  What we want to improve
- Many parameters are obtained from config file in shell script, since `gcloud` command does not work as expected. But I think it is not beautiful.




