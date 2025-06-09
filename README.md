# Roast Analysis

A professional Python package for analyzing coffee roast curves from Artisan .alog files. This package provides tools for parsing, processing, scoring, and visualizing roast data to help improve coffee roasting consistency and quality.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy-lang.org/)

## Features

- **Parse .alog files**: Extract temperature, timing, and RoR data from Artisan log files
- **Data processing**: Clean outliers, normalize curves, and segment roast phases  
- **Quality scoring**: Score roasts based on RoR characteristics, timing, and temperature profiles
- **Visualizations**: Generate plots for individual roasts and comparative analysis
- **CLI tool**: Command-line interface for batch processing
- **Type safety**: Full type annotations and mypy checking
- **Test coverage**: pytest-based test suite
- **Code quality**: ruff linting and formatting

## Installation

### Development Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd roast-curve-analysis
```

2. **Create and activate conda environment:**
```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda create -n roast-analysis python=3.11 -y
conda activate roast-analysis
```

3. **Install the package in development mode:**
```bash
pip install -e ".[dev]"
```

4. **Install pre-commit hooks:**
```bash
pre-commit install
```

### Production Installation

```bash
pip install roast-analysis
```

## Quick Start

### Command Line Interface

```bash
# Analyze all .alog files in the data/ folder
roast-analyze

# Analyze files in a specific folder and export to CSV
roast-analyze /path/to/roast/logs --export-csv
```

### Python API

```python
from roast_analysis import load_roast_data_as_dataframe, process_roast_data

# Load roast data
df = load_roast_data_as_dataframe("data/")

# Process and analyze
processed_df = process_roast_data(df)

print(f"Processed {len(processed_df)} roasts")
```

## Project Structure

```
roast-curve-analysis/
├── src/roast_analysis/        # Main package code
│   ├── __init__.py           # Package exports
│   ├── alog_parser.py        # Parse .alog files
│   ├── data_processor.py     # Clean and process data
│   ├── scoring_engine.py     # Calculate quality scores
│   ├── roast_plotter.py      # Visualization functions
│   ├── cli.py               # Command-line interface
│   └── py.typed             # Type checking marker
├── tests/                   # Test suite
│   ├── test_alog_parser.py
│   └── test_data_processor.py
├── docs/                    # Documentation
├── pyproject.toml          # Project configuration
├── .pre-commit-config.yaml # Pre-commit hooks
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## Development

### Code Quality Tools

This project uses modern Python development tools:

- **ruff**: Lightning-fast linting and formatting
- **mypy**: Static type checking
- **pytest**: Testing framework with coverage
- **pre-commit**: Git hooks for code quality

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src/roast_analysis --cov-report=html

# Run specific test file
pytest tests/test_alog_parser.py -v
```

### Linting and Formatting

```bash
# Check code quality
ruff check src/

# Format code
ruff format src/

# Type checking
mypy src/
```

### Pre-commit Hooks

Pre-commit hooks automatically run on each commit to ensure code quality:

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## API Reference

### Core Functions

#### `load_roast_data_as_dataframe(folder: Path) -> pd.DataFrame`
Load multiple .alog files into a pandas DataFrame.

#### `process_roast_data(df: pd.DataFrame, filter_data: bool = True) -> pd.DataFrame`
Complete processing pipeline including cleaning, segmentation, and normalization.

#### `score_roast_dataframe(df: pd.DataFrame) -> pd.DataFrame`
Calculate quality scores for all roasts in the DataFrame.

### Data Processing Pipeline

1. **Parse .alog files** → Extract raw temperature and timing data
2. **Annotate events** → Validate and timestamp roast events  
3. **Filter invalid roasts** → Remove incomplete or unrealistic data
4. **Segment curves** → Extract charge-to-drop periods
5. **Clean data** → Remove outliers using Hampel filter
6. **Normalize** → Resample to consistent time scale (0→1)
7. **Calculate metrics** → Development ratios, phase timing
8. **Score quality** → Multi-component scoring system

## Understanding Scores

### Composite Score (0-100)
Weighted combination of four components:
- **RoR Score (40%)**: Rate of rise curve characteristics
- **Timing Score (20%)**: Duration vs target (10.5 minutes)  
- **Temperature Score (20%)**: Drop temperature vs target (388°F)
- **Development Score (20%)**: Development ratio vs target (22.5%)

### Score Categories
- **Excellent**: 80-100 points
- **Good**: 60-79 points
- **Fair**: 40-59 points  
- **Poor**: 0-39 points

## Configuration

The project uses `pyproject.toml` for all configuration:

- **Build system**: setuptools with PEP 621 metadata
- **Dependencies**: Production and development requirements
- **Tool configuration**: ruff, pytest, mypy, coverage settings
- **Entry points**: CLI command registration

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes with proper tests and type annotations
4. Ensure all checks pass: `pre-commit run --all-files`
5. Submit a pull request

### Development Guidelines

- Follow type hints throughout the codebase
- Write tests for new functionality
- Update documentation for API changes
- Use descriptive commit messages
- Ensure ruff and mypy checks pass

## Data Requirements

### .alog File Format
Each .alog file must contain:
- Temperature data: `temp1` (ET), `temp2` (BT)
- Time data: `timex`
- Event markers: CHARGE, TP, FCs, DROP
- Event temperatures for validation

### Quality Filters
Automatic filtering removes roasts with:
- Duration < 5 minutes or > 20 minutes
- Drop temperature < 350°F or > 450°F
- Missing charge time data
- Incomplete event markers

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
- **Bug reports**: GitHub Issues
- **Documentation**: README and inline docstrings
- **Contributing**: See CONTRIBUTING.md