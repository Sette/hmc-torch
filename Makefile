
VERSION="0.0.8"

lint-check:
	@echo "--> Running linter check"
	autopep8 --in-place --recursive src
	flake8 **/*.py
	black --check **/*.py
	pylama **/*.py
	isort -c **/*.py
	pylint **/*.py

pre-commit:
	@echo "--> Running pre-commit"
	pre-commit run --all-files

lint:
	@echo "--> Running linter"
	black **/*.py
	flake8 **/*.py
	isort **/*.py
	pylint **/*.py

dvc:
	@dvc pull

test:
	@echo "--> Running Test"
	@pytest --verbose --cov-report term-missing --cov-report xml --cov-report html --cov=. .
	@echo ""

build:
	@echo "--> Docker build"
	docker build -f Dockerfile -t hmc-torch:$(VERSION) .