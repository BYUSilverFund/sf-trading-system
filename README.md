# Silver Fund Trading System Repository
The all in one repository for researching, backtesting, and trading quantitative trading strategies.

## Getting Started

### Install UV
MacOS
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows
```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Sync Denpendencies
```bash
uv sync
```

### Activate Environment
MacOS
```bash
source .venv/bin/activate
```

Windows
```bash
.venv/Scripts/activate
```

### Add Module to Python Path
I prefer editing rc files in vs code but you can also use vim or nano
Note: if you don'
```bash
code ~/.zshrc
```

Add this line to the bottom:
```bash
# Custom python path for quant-fabric project
export PYTHONPATH=/path/to/project/sf-trading-system/:$PYTHONPATH
```

Make sure to kill all running terminals and reopen for the environment to reset.

### Install pre-commit
``` bash
pre-commit install
```

That's it! You are all ready to go with the project!

## Commands

### Add a Dependency
```bash
uv add <package>
```

## Remove a Dependency
```bash
uv remove <package>
```

## Format Code
```bash
pre-commit run --all-files
```
