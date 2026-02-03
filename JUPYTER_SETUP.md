# Jupyter Environment Setup

This guide helps you set up a Jupyter notebook environment for exploring the M3Benchmark dataset.

## Quick Start

### 1. Create a virtual environment

```bash
cd /Users/anu/Documents/GitHub/routing/EnterpriseBenchmark
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Register the kernel (optional, for VSCode/JupyterLab)

```bash
python -m ipykernel install --user --name=m3benchmark --display-name="M3Benchmark"
```

### 4. Launch Jupyter

**JupyterLab (recommended):**
```bash
jupyter lab
```

**Classic Notebook:**
```bash
jupyter notebook
```

**VSCode:**
- Open any `.ipynb` file
- Select the `m3benchmark` kernel from the kernel picker

## Included Packages

| Package | Purpose |
|---------|---------|
| `jupyter` / `jupyterlab` | Notebook interface |
| `pandas` | Data manipulation and analysis |
| `numpy` | Numerical computing |
| `matplotlib` / `seaborn` | Data visualization |
| `jsonlines` | Reading JSONL files |
| `tqdm` | Progress bars |

## Directory Structure

```
EnterpriseBenchmark/
├── train_chunked/       # Training data (JSON files)
├── sample_data/         # Sample data for testing
├── apis/                # API definitions
├── evaluation.py        # Evaluation script
├── notebooks/           # Your Jupyter notebooks (create this)
└── requirements.txt     # Python dependencies
```

## Example: Loading Data

```python
import json
import pandas as pd

# Load a dataset
with open('train_chunked/shakespeare_multiturn_bird_chunked.json', 'r') as f:
    data = json.load(f)

# Explore structure
print(f"Number of samples: {len(data)}")
print(f"Keys: {data[0].keys() if isinstance(data, list) else data.keys()}")
```

## Troubleshooting

**Kernel not found:**
```bash
python -m ipykernel install --user --name=m3benchmark
```

**Import errors:**
```bash
pip install -r requirements.txt --force-reinstall
```

**Permission issues:**
```bash
pip install --user -r requirements.txt
```
