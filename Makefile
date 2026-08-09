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
	uv run flake8 src/
	uv run ruff format --check src/
	uv run ruff check src/
	uv run pylint $$(git ls-files '*.py' | grep -v '^experiments/' | grep -v '^tests/' | grep -v '^notebooks/')

pre-commit:
	@echo "--> Running pre-commit"
	uv run pre-commit run --all-files

lint:
	@echo "--> Running linter"
	uv run flake8 src/
	uv run ruff format src/
	uv run ruff check src/
	uv run pylint $$(git ls-files '*.py' | grep -v '^experiments/' | grep -v '^tests/'| grep -v '^notebooks/')

run:
	./run.sh --device cuda --dataset_name wos --method global --output_path output

test:
	@echo "--> Running Test"
	@pytest --verbose --cov-report term-missing --cov-report xml --cov-report html --cov=. .
	@echo ""

build:
	@echo "--> Docker build"
	docker build -f Dockerfile -t hmc-torch:$(VERSION) .
