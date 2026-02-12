
# CC-IFPS: Class-Conditioned Irredundant Feature Patch Selection (Proposed Method)

This repository contains the implementation of the **Class-Conditioned Irredundant Feature Patch Selection (CC-IFPS)** algorithm, a hybrid anomaly detection method combining Greedy Coreset Sampling and D2-Sampling for optimal memory efficiency and high performance.

## Structure

- `src/`: Core source code for PatchCore and the proposed CC-IFPS sampler.
- `bin/`: Executable scripts (e.g., `run_patchcore.py`).
- `docs/`: Documentation and experimental logs.
- `utils/`: Helper scripts for validation (memory efficiency, variance checks).
- `results/`: Directory for experiment outputs.
- `run_experiment.sh`: Main script to reproduce the "Golden Configuration" results.
- `analyze_results.py`: Script to analyze and summarize experiment results.

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   (Recommended: Use a virtual environment like Conda)

2. Install the package:
   ```bash
   pip install -e .
   ```

## Usage

### verify "Golden Configuration" (Reproduction)
To run the verified optimal configuration for all MVTec AD classes:

```bash
./run_experiment.sh
```
This script will:
- Execute the experiment for all 15 classes using the best sampler (Greedy or D2) and hyperparameters confirmed in our extensive testing.
- Run Seed 1 for quick validation (modify script to run multiple seeds if needed).
- Automatically analyze and display the results upon completion.

### Analyze Results manually
If you want to re-analyze existing results:

```bash
python analyze_results.py
```

### Validation Tools (in `utils/`)
- `compare_memory_efficiency.py`: Compares memory usage between Proposed and Baseline methods.
- `run_variance_check.sh`: Verifies stability across multiple seeds.
- `run_memory_efficiency_check.sh`: Runs a dedicated memory benchmark.

Example:
```bash
python utils/compare_memory_efficiency.py
```

## Dataset
Please place the MVTec AD dataset in a `data/` directory or update the path in `run_experiment.sh`.

## License
MIT License
