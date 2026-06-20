
VERSION="0.0.8"

export PYTHONPATH=src

# ── Datasets ────────────────────────────────────────────────────────
# Download individual datasets

download-arxiv:
	python -m hmc.datasets.arxiv.download_arxiv --output_dir ./data

download-wos:
	python -m hmc.datasets.wos.download_wos --output_dir ./data/wos

download-arff-fun:
	python -m hmc.datasets.gofun.download_arff --output_dir ./data --subset FUN

download-arff-go:
	python -m hmc.datasets.gofun.download_arff --output_dir ./data --subset GO

download-arff-others:
	python -m hmc.datasets.gofun.download_arff --output_dir ./data --subset others

download-all:
	python -m hmc.datasets.download_all --continue-on-error

# ── Lint / Test / Run ───────────────────────────────────────────────

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
	pylint $$(git ls-files '*.py')

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
