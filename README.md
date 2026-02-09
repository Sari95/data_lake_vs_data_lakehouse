# data_lake_vs_data_lakehouse
A minimal Python example that demonstrates how mutable data references in traditional data-lake setups can make model results harder to explain and reproduce.
This repository contains a small, deterministic ML demo showing the difference between:
- Data Lake runs: Same file path, data overwritten
- Data Lakehouse runs: Explicit data versions

The example uses a synthetic churn dataset and logistic regression to highlight a common issue: Similar model accuracy does not guarantee reproducibility if the underlying data changes.

# Run the Demo
python -m venv .venv
.\.venv\Scripts\activate   # Windows
pip install pandas numpy scikit-learn duckdb

python scripts/run_demo.py

# After running:
- data_lake/ contains the mutable dataset
- lakehouse/ contains versioned datasets
- scripts/runs_meta/ stores metadata for each run
