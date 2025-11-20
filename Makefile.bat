@echo off
REM Makefile for Windows - Conversion of Python tasks
REM Usage: build.bat [target]
REM Example: build.bat install-dependencies

setlocal enabledelayedexpansion

if "%1"=="" (
    echo.
    echo Available targets:
    echo   install-dependencies  - Install Python dependencies
    echo   lint-check            - Run linter check (no modifications)
    echo   pre-commit            - Run pre-commit hooks
    echo   lint                  - Run linters (with modifications)
    echo   dvc                   - Pull data using DVC
    echo   test                  - Run tests with coverage
    echo   build                 - Build Docker image
    echo.
    exit /b 0
)
REM Install dependencies
if "%1"=="install-dependencies" (
    echo.
    echo --> Installing Python dependencies
    pip --disable-pip-version-check install -r requirements.txt
    echo.
    exit /b 0
)

REM Lint check
if "%1"=="lint-check" (
    echo.
    echo --> Running linter check
    autopep8 --in-place --recursive src
    flake8 .
    black --check .
    pylama .
    isort -c .
    pylint src
    exit /b 0
)

REM Pre-commit
if "%1"=="pre-commit" (
    echo.
    echo --> Running pre-commit
    pre-commit run --all-files
    exit /b 0
)

REM Lint
if "%1"=="lint" (
    echo.
    echo --> Running linter
    black .
    flake8 .
    isort .
    exit /b 0
)

REM DVC
if "%1"=="dvc" (
    dvc pull
    exit /b 0
)

REM Test
if "%1"=="test" (
    echo.
    echo --> Running Test
    poetry run pytest --verbose --cov-report term-missing --cov-report xml --cov-report html --cov=. .
    echo.
    exit /b 0
)

REM Build
if "%1"=="build" (
    echo.
    echo --> Docker build
    docker build -f Dockerfile -t hmc-torch:0.0.1 .
    exit /b 0
)

echo Unknown target: %1
exit /b 1
