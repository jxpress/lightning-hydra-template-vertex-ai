
help:  ## Show help
	@grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

clean: ## Clean autogenerated files
	rm -rf dist
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -f .coverage

clean-logs: ## Clean logs
	rm -r logs/**

style: ## Run pre-commit hooks
	pre-commit run -a

sync: ## Merge changes from main branch to your current branch
	git fetch --all
	git merge main

test: ## Run not slow tests
	pytest -k "not slow"

test-full: ## Run all tests
	pytest

train: ## Train the model
	python src/train.py

debug: ## Enter debugging mode with pdb
	#
	# tips:
	# - use "import pdb; pdb.set_trace()" to set breakpoint
	# - use "h" to print all commands
	# - use "n" to execute the next line
	# - use "c" to run until the breakpoint is hit
	# - use "l" to print src code around current line, "ll" for full function code
	# - docs: https://docs.python.org/3/library/pdb.html
	#
	python -m pdb src/train.py debug=default

train-in-docker: ## Run train.py in docker Image
	docker-compose up


docker-push: ## build docker image and push it to GCP. you have to change <uri of GCP artifact repository> below
	docker build . -t <uri of GCP artifact repository>
	docker push <uri of GCP artifact repository>

create-custom-job: ## create custom job of Vertex AI
	chmod +x ./vertex_ai/scripts/custom_job/create_job.sh
	./vertex_ai/scripts/custom_job/create_job.sh
	
create-hparams-tuning-job: ## create hyperparameter tuning job of Vertex AI
	chmod +x ./vertex_ai/scripts/hparams_tuning_job/create_job.sh
	./vertex_ai/scripts/hparams_tuning_job/create_job.sh