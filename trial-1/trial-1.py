from huggingface_hub import login
import os

# Environment-level MPS suppression
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# Set HuggingFace cache directory to local `../.cache`
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.path.dirname(__file__), "../.cache")
os.environ["HF_DATASETS_CACHE"] = os.path.join(os.path.dirname(__file__), "../.cache")

login(token=os.getenv("HUGGINGFACE_TOKEN"))

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
from transformers import Trainer
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import random
import json
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Fix seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# Force CPU input conversion inside Trainer
class CPUTrainer(Trainer):
    def training_step(self, model, inputs):
        model = model.to("cpu")
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        return super().training_step(model, inputs)

    def evaluation_loop(self, *args, **kwargs):
        self.model.to("cpu")
        return super().evaluation_loop(*args, **kwargs)

def plot_scaled_parameters(log_data):
    if not log_data:
        return

    generations = [entry["generation"] for entry in log_data]
    fitness = [entry["fitness"] for entry in log_data]
    lr = [entry["learning_rate"] for entry in log_data]
    dropout = [entry["dropout"] for entry in log_data]
    batch_size = [entry["batch_size"] for entry in log_data]

    # Normalize to [0,1]
    def scale(values):
        min_val = min(values)
        max_val = max(values)
        return [(v - min_val) / (max_val - min_val + 1e-8) for v in values]

    fitness_scaled = scale(fitness)
    lr_scaled = scale(lr)
    dropout_scaled = scale(dropout)
    batch_scaled = scale(batch_size)

    plt.clf()
    plt.plot(generations, fitness_scaled, label="Fitness", color='green')
    plt.plot(generations, lr_scaled, label="Learning Rate", linestyle="--")
    plt.plot(generations, dropout_scaled, label="Dropout", linestyle="-.")
    plt.plot(generations, batch_scaled, label="Batch Size", linestyle=":")

    best_gen = max(log_data, key=lambda x: x["fitness"])["generation"]
    plt.axvline(x=best_gen, color='gray', linestyle='--', alpha=0.4)

    plt.title("Scaled Hyperparameters Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Scaled Value")
    plt.legend(loc="upper right")
    plt.pause(0.05)

def main():
    set_seed()

    print("MPS disabled. Using CPU.")
    plt.ion()
    plt.figure(figsize=(10, 6))
    plt.show(block=False)
    plt.plot([0], [0], alpha=0.0)
    plt.pause(0.05)

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return {"accuracy": accuracy_score(p.label_ids, preds)}

    # Load dataset
    raw_datasets = load_dataset("glue", "sst2")
    model_name = "google/mobilebert-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Hyperparameter bounds
    lr_bounds = (1e-6, 1e-3)
    dropout_bounds = (0.1, 0.5)
    batch_sizes = [16, 32, 64]

    theta = {
        "learning_rate": np.random.uniform(*lr_bounds),
        "dropout": np.random.uniform(*dropout_bounds),
        "batch_size": int(np.random.choice(batch_sizes))
    }

    sigma_m = 0.1
    success_count = 0
    mutation_trials = 0

    best_fitness = float("inf")
    best_theta = theta.copy()

    def evaluate_config(theta, log_data, gen):
        torch.manual_seed(42)

        with torch.device("cpu"):
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=2
            ).to("cpu")

        args = TrainingArguments(
            output_dir="./results",
            eval_strategy="epoch",
            learning_rate=theta["learning_rate"],
            per_device_train_batch_size=theta["batch_size"],
            per_device_eval_batch_size=theta["batch_size"],
            num_train_epochs=1,
            logging_steps=10,
            save_strategy="no",
            seed=42,
            disable_tqdm=True
        )

        trainer = CPUTrainer(
            model=model,
            args=args,
            train_dataset=tokenized_datasets["train"].shuffle(seed=42).select(range(2000)),
            eval_dataset=tokenized_datasets["validation"].select(range(500)),
            compute_metrics=compute_metrics
        )

        trainer.train()
        metrics = trainer.evaluate()

        log_data.append({
            "generation": gen,
            "fitness": -metrics["eval_accuracy"],
            "learning_rate": theta["learning_rate"],
            "dropout": theta["dropout"],
            "batch_size": theta["batch_size"],
            "improved": False
        })

        plot_scaled_parameters(log_data)
        return -metrics["eval_accuracy"]

    max_generations = 50
    patience = 10
    no_improvement_counter = 0

    gen = 0
    log_data = []

    while gen < max_generations and no_improvement_counter < patience:
        theta_prime = {
            "learning_rate": np.clip(theta["learning_rate"] + sigma_m * np.random.randn() * 1e-4, *lr_bounds),
            "dropout": np.clip(theta["dropout"] + sigma_m * np.random.randn() * 0.05, *dropout_bounds),
            "batch_size": int(np.random.choice(batch_sizes))
        }

        fitness = evaluate_config(theta_prime, log_data, gen)
        mutation_trials += 1

        if fitness < best_fitness:
            best_fitness = fitness
            best_theta = theta_prime.copy()
            theta = theta_prime
            success_count += 1
            no_improvement_counter = 0
            log_data[-1]["improved"] = True
            plot_scaled_parameters(log_data)
        else:
            no_improvement_counter += 1
            plot_scaled_parameters(log_data)

        if mutation_trials % 5 == 0:
            success_rate = success_count / mutation_trials
            sigma_m *= 1.2 if success_rate > 0.2 else 0.82
            success_count = 0
            mutation_trials = 0

        gen += 1

    with open("meta_es_log.json", "w") as f:
        json.dump(log_data, f, indent=2)

    print("Best Hyperparameters:", best_theta)
    print("Best Validation Accuracy:", -best_fitness)
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
