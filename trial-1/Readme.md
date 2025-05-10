# README
# Hyperparameter Optimization with Meta-Evolution Strategies (Meta-ES)

## Abstract

This project implements a Meta-Evolution Strategy (Meta-ES) to optimize the hyperparameters of a transformer-based text classification model on the SST-2 sentiment analysis task from the GLUE benchmark. The Meta-ES algorithm adapts learning rate, dropout rate, and batch size across generations to maximize validation accuracy. The code includes real-time visualization of optimization progress and supports deployment on constrained environments. The best configuration identified achieved a validation accuracy of 81% using the lightweight `google/mobilebert-uncased` model.

## Introduction

Hyperparameter optimization is a critical step in training neural models, especially transformer-based architectures, where parameters like learning rate and dropout significantly affect generalization. Traditional grid and random search approaches can be inefficient, particularly when constrained by limited compute resources.

To address this, we implement a (1+1) Meta-Evolution Strategy inspired by the work of Kramer[1], where an outer evolutionary algorithm mutates hyperparameters and evaluates their fitness through an inner optimizer loop. The model under evaluation is a MobileBERT variant, chosen for its efficiency on CPU-only systems.

## Methods

### Dataset

* **Source**: GLUE benchmark
* **Task**: SST-2 (Stanford Sentiment Treebank v2)
* **Samples Used**:

  * Training subset: 2,000 examples
  * Validation subset: 500 examples

### Model

* **Base Model**: `google/mobilebert-uncased`
* **Head**: Sequence classification head with two labels (positive/negative)
* **Framework**: HuggingFace Transformers with PyTorch backend

### Hyperparameters

The Meta-ES optimizes the following:

* **Learning Rate**: \[1e-6, 1e-3]
* **Dropout**: \[0.1, 0.5]
* **Batch Size**: \[16, 32, 64]

### Optimization Loop

* **Outer loop**: Meta-Evolution Strategy ((1+1)-ES)
* **Mutation**: Gaussian perturbation with dynamic mutation strength (`sigma_m`)
* **Adaptation Rule**: Rechenberg's 1/5th success rule
* **Early Stopping**: Patience-based (10 generations without improvement)

### Evaluation Metric

* **Fitness Function**: Negative validation accuracy (to convert to minimization)

### Visualization

* Real-time plotting of scaled hyperparameters and fitness across generations
* Live updates after every training epoch and generation

## Results

* **Best Configuration Found**:

```json
{
  "learning_rate": 0.0003799,
  "dropout": 0.4871,
  "batch_size": 32
}
```

* **Validation Accuracy**: **81%**
* **Training Runtime per Generation**: \~107 seconds
* **Eval Loss Range**: 0.39 to 0.52 across configurations

## Discussion

This prototype demonstrates that Meta-Evolution Strategies can be used effectively for low-resource hyperparameter tuning in transformer-based NLP tasks. By combining a small validation subset and a compact model (MobileBERT), we achieved meaningful accuracy while keeping evaluation times reasonable.

The optimization trajectory showed that good configurations could be found early, but stagnation occurred due to a narrow mutation space and the absence of diversity or restart mechanisms. Grad norm patterns and learning rate decay aligned with typical training dynamics.

This setup is CPU-friendly and ready for porting to Jetson Nano or similar ARM-based edge devices, provided models are adapted or frozen as needed. Future directions include integrating warm starts, weight freezing, and alternate fitness objectives (e.g., F1-score or loss).

---

**Authors**: Project implemented by Susanne Coates with support from HuggingFace Transformers and the GLUE benchmark dataset.

**License**: MIT or Apache 2.0 depending on deployment context.

## References
 1. Kramer, O. (2025). Enhancing Evolutionary Algorithms through Meta-Evolution Strategies. In *Proceedings of the 2025 IEEE Conference on Artificial Intelligence (CAI)* IEEE. ISBN: 979-8-3315-2400-5

