# Yildiz

A modular framework for **astrophysics time-series data ingestion, preprocessing, and model orchestration**.

The goal is to build a reproducible system that:
- ingests heterogeneous data sources
- standardizes and preprocesses inputs
- routes data through statistical / AI models
- enables consistent experimentation and analysis

```
yildiz/
├── yildiz/
│   ├── __init__.py
│   ├── module.py
│   ├── pipeline.py
│   ├── preprocessing.py
|   ├── segmentation.py
│   └── io/
│       ├── __init__.py
│       ├── base.py
│       ├── local.py
│       ├── registry.py
│       └── remote.py
│   └── analysis/
|       ├── comparison.py
│       ├── component_analysis.py
│       ├── w_correlation.py
│   └── preprocessors/
|       ├── prewhitening.py
│       ├── ssa.py
│       ├── fft.py
|   └── results/
|       ├── __init__.py
|       ├── prewhitening_result.py
|       ├── fft_result.py
|       ├── ssa_result.py
│   └── utils/
│       ├── performance.py
├── tests/
│   └── ...
├── LICENSE
└── README.md
```

## Design Principles

- **Modularity**: every component is isolated and reusable  
- **Separation of concerns**:
  - `io/` handles external data
  - `yildiz/` handles processing and orchestration  

---

## Current Scope

- Basic module abstraction  
- Pipeline execution system  
- Initial data ingestion layer (`io/`)
- Preparing the data for preprocessing methods (separating into continuous chunks, etc.)
- FFT, SSA preprocessing after segmentation to continuous chunks

---

## Example (conceptual)

```python
from yildiz.io.registry import Registry # Dataset pointers/reference to access local database
from yildiz.io.local import LocalFITS

import matplotlib.pyplot as plt

registry = Registry()

# Inspect available products
for row in registry.list_products():

    print(row)

# Select a product from database
product_id = 1

fits_path = registry.get_filepath(product_id)

print()
print("Selected file:")
print(fits_path)

# Load through astrocore abstraction
source = LocalFITS(fits_path)

lc = source.load()

# Quick visualization
plt.figure()

plt.scatter(
    lc["t"],
    lc["y"],
    s=1
)

plt.xlabel("TIME")
plt.ylabel("PDCSAP_FLUX")
plt.tight_layout()

plt.show()
```

## Notes

This repository is under active development.
Structure, naming, and interfaces are expected to evolve.
