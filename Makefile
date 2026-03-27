
VERSION="0.0.7"

lint-check:
	@echo "--> Running linter check"
	autopep8 --in-place --recursive src
	flake8 **/*.py
	black --check **/*.py
	ruff ckeck .
	isort -c **/*.py
	pylint **/*.py

pre-commit:
	@echo "--> Running pre-commit"
	pre-commit run --all-files

lint:
	@echo "--> Running linter"
	autopep8 --in-place --recursive src
	flake8 **/*.py
	black **/*.py
	ruff format **/*.py
	ruff check . --fix
	isort **/*.py
	pylint **/*.py

dvc:
	@dvc pull

run:
	./run.sh --device cuda --dataset_name seq_FUN --output_path output --method local --epochs_to_evaluate 10 

run_tabat:
	./run.sh --device cuda --dataset_name seq_FUN --output_path output --method local_tabat --epochs_to_evaluate 10 

test:
	@echo "--> Running Test"
	@pytest --verbose --cov-report term-missing --cov-report xml --cov-report html --cov=. .
	@echo ""

build:
	@echo "--> Docker build"
	docker build -f Dockerfile -t hmc-torch:$(VERSION) .