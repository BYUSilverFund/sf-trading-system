# Silver Fund Trading System Repository
The all in one repository for researching, backtesting, and trading quantitative trading strategies.

# Getting Started

## Clone Repository
In your desired folder run:
```bash
git clone https://github.com/BYUSilverFund/sf-trading-system.git
```

Open in vscode
```
code sf-trading-system
```

## Git Setup
Configure your git credentials. Otherwise you won't be able to commit any code.

```bash
git config --global user.name <username>
```
```bash
git config --global user.email <email>
```

## Virtual Environment Setup
The virtual environment will make it so that we have consistent package and Python versions across all of our devices running the repository.

### `uv` Installation (Package Manager)

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
source $HOME/.local/bin/env
```
Restart your terminal for the changes to take effect.

### Sync Denpendencies
```bash
uv sync
```

### Activate Environment
MacOS/Linux (Supercomputer)
```bash
source .venv/bin/activate
```

Windows
```bash
.venv/Scripts/activate
```

## Add Module to Python Path
This step will allow for you to import any module from the project to any other file.

```bash
code ~/.bashrc
```

Add this line to the bottom:
```bash
# Custom python path for sf-trading-system project
export PYTHONPATH=/path/to/project/sf-trading-system/:$PYTHONPATH
```

Make sure to kill all running terminals and reopen for the environment to reset.

## Install `pre-commit`
pre-commit will clean up any code that you write when you commit it. This helps with formatting consistency across the repository.
``` bash
pre-commit install
```

## Create `.env` File
Create a new file in your `sf-trading-system` directory called `.env`.
In the file add the following line but with your path.

```bash
ROOT=/path/to/sf-trading-system
```

That's it! You are all ready to go with the project!

## Useful Commands

### Add a Dependency
```bash
uv add <package>
```

### Remove a Dependency
```bash
uv remove <package>
```

### Format Code
```bash
pre-commit run --all-files
```
