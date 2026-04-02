
VERSION="0.0.7"

lint-check:
	@echo "--> Running linter check"
	autopep8 --in-place --recursive src
	flake8 src/
	black --check src/
	ruff check src/
	isort -c src/
	pylint src/

pre-commit:
	@echo "--> Running pre-commit"
	pre-commit run --all-files

lint:
	@echo "--> Running linter"
	autopep8 --in-place --recursive src
	flake8 src/
	black src/
	ruff format src/
	ruff check src/ --fix
	isort src/
	pylint src/

dvc:
	@dvc pull

run:
	./run.sh --device cuda --dataset_name seq_FUN --output_path output --method local --epochs_to_evaluate 10 

test:
	@echo "--> Running Test"
	@pytest --verbose --cov-report term-missing --cov-report xml --cov-report html --cov=. .
	@echo ""

build:
	@echo "--> Docker build"
	docker build -f Dockerfile -t hmc-torch:$(VERSION) .