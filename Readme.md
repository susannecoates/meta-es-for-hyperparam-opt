# README: Installation and Usage Guide

This project implements a Meta-Evolution Strategy (Meta-ES) to optimize hyperparameters for a transformer-based sentiment classification model on the SST-2 dataset using Hugging Face Transformers.

---

## Installation

### 1. Clone the repository

```bash
git clone git@github.com:susannecoates/meta-es-for-hyperparam-opt.git
cd meta-es-for-hyperparam-opt
```

### 2. Set up a virtual environment (optional but recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, manually install:

```bash
pip install torch torchvision
pip install transformers datasets
pip install scikit-learn matplotlib
pip install huggingface-hub
```

> Note: On Jetson or ARM-based systems, install PyTorch using NVIDIA's JetPack SDK instructions.

---

## Hugging Face Token (Required for some models)

Create a `.env` file or export your token:

```bash
export HUGGINGFACE_TOKEN=your_hf_token_here
```

Or use:

```python
from huggingface_hub import login
login(token="your_hf_token")
```

---

## Running the Program

For example, to launch the hyperparameter search with Meta-ES for the 1st experiment:

```bash
cd experiment-1
python3 experiment-1.py
```

During execution:

* A live matplotlib window shows optimization progress
* The best hyperparameters and validation accuracy are printed at the end
* All generation logs are saved to `meta_es_log.json`

---

## Requirements Summary

* Python 3.8+
* PyTorch (with CUDA optional)
* Transformers
* Datasets
* scikit-learn
* matplotlib
* huggingface-hub

---

## 🔧 Troubleshooting

* If no plot window appears, verify your matplotlib backend (TkAgg works well on most systems).
* For headless systems, use `matplotlib.use("Agg")` and save plots to file.
* If models fail to load, verify your Hugging Face token and internet access.

---

## License

MIT License. See `LICENSE` file for details.

---

## Author

The work in this project was implemented by Dr. Susanne Coates and is based on the conference papers, "Enhancing Evolutionary Algorithms through Meta-Evolution Strategies," and "Evolutionary Cognitive Prompting for Enhancing the Capabilities of Language Models," both by Prof. Dr. Oliver Kramer, Carl v. Ossietzky University of Oldenburg, Germany.