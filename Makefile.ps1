# Build script for Windows - PowerShell Version
# Usage: .\Makefile.ps1 install-dependencies
# Or:    .\Makefile.ps1 -Task install-dependencies

param(
    [Parameter(Position = 0)]
    [string]$Task = 'help'
)

# Task mapping
$tasks = @{
    'install-dependencies' = 'InstallDependencies'
    'lint-check' = 'LintCheck'
    'pre-commit' = 'PreCommit'
    'lint' = 'Lint'
    'dvc' = 'DVCPull'
    'test' = 'RunTests'
    'build' = 'BuildDocker'
    'help' = 'ShowHelp'
}

function ShowHelp {
    Write-Host "`nAvailable targets:`n" -ForegroundColor Cyan
    Write-Host "  install-dependencies  - Install Python dependencies"
    Write-Host "  lint-check            - Check linting (without modifications)"
    Write-Host "  pre-commit            - Run pre-commit hooks"
    Write-Host "  lint                  - Run linting with modifications"
    Write-Host "  dvc                   - Pull DVC data"
    Write-Host "  test                  - Run tests with coverage"
    Write-Host "  build                 - Build Docker image`n"
}

function InstallDependencies {
    Write-Host "--> Installing Python dependencies" -ForegroundColor Cyan
    pip --disable-pip-version-check install -r requirements.txt
    Write-Host ""
}

function LintCheck {
    Write-Host "--> Running linter check" -ForegroundColor Cyan
    autopep8 --in-place --recursive src
    flake8 .
    black --check .
    pylama .
    isort -c .
    pylint src
}

function PreCommit {
    Write-Host "--> Running pre-commit" -ForegroundColor Cyan
    pre-commit run --all-files
}

function Lint {
    Write-Host "--> Running linter" -ForegroundColor Cyan
    black .
    flake8 .
    isort .
}

function DVCPull {
    Write-Host "--> Pulling DVC data" -ForegroundColor Cyan
    dvc pull
}

function RunTests {
    Write-Host "--> Running tests" -ForegroundColor Cyan
    poetry run pytest --verbose --cov-report xml --cov-report html --cov=. .
    Write-Host ""
}

function BuildDocker {
    Write-Host "--> Docker build" -ForegroundColor Cyan
    docker build -f Dockerfile -t hmc-torch:0.0.1 .
}

# Execute task
if ($tasks.ContainsKey($Task)) {
    & $tasks[$Task]
} else {
    Write-Host "Unknown task: $Task" -ForegroundColor Red
    ShowHelp
    exit 1
}
