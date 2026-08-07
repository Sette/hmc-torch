VERSION="0.0.9"

export PYTHONPATH=src

# ── Datasets ────────────────────────────────────────────────────────

download-arxiv:
	python -m hmc.datasets.arxiv.download_arxiv --output_dir ./data

download-wos:
	python -m hmc.datasets.wos.download_wos --output_dir ./data/wos

download-aapd:
	python -m hmc.datasets.aapd.download_aapd --output_dir ./data/aapd

download-rcv1:
	python -m hmc.datasets.rcv1.download_rcv1 --output_dir ./data/rcv1

download-eurlex:
	python -m hmc.datasets.eurlex.download_eurlex --output_dir ./data/eurlex

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

run:
	./run.sh --device cuda --dataset_name wos --method global --output_path output

test:
	@echo "--> Running Test"
	@pytest --verbose --cov-report term-missing --cov-report xml --cov-report html --cov=. .
	@echo ""

build:
	@echo "--> Docker build"
	docker build -f Dockerfile -t hmc-torch:$(VERSION) .
