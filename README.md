# Silver Fund Trading System Repository
The all in one repository for researching, backtesting, and trading quantitative trading strategies.

## Getting Started

### Clone Repository
In your desired folder run
```bash
git clone https://github.com/BYUSilverFund/sf-trading-system.git
```

Open in vscode
```
code sf-trading-system
```

### Setup git
```bash
git config --global user.name <username>
```
```bash
git config --global user.email <email>
```

### Install UV
MacOS/Linux (Supercomputer)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows
```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Check installation
```bash
uv --version
```

If this returns an error you might need to add uv to your path. Run:
```bash
code ~/.bashrc
```

And add the following to the bottom of the file.
```bash
# UV
. "$HOME/.local/bin/env"
```
Restart your terminal for the changes to take effect.

### Sync Denpendencies
```bash
uv sync
```

### Activate Environment
MacOS/Linux
```bash
source .venv/bin/activate
```

Windows
```bash
.venv/Scripts/activate
```

### Add Module to Python Path
```bash
code ~/.bashrc
```

Add this line to the bottom:
```bash
# Custom python path for sf-trading-system project
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
