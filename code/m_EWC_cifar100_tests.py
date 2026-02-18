# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 09:20:49 2026

@author: maboorai279
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18

from avalanche.benchmarks.classic import SplitCIFAR100
from avalanche.training.supervised import EWC
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger

import csv
import os
from datetime import datetime
import random
import numpy as np


# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================
# SINGLE EXPERIMENT
# ============================================================
def run_experiment(config):
    
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Benchmark
    benchmark = SplitCIFAR100(
        n_experiences = 10,       # 10 experiences × 10 classes
        return_task_id=False,     # Single-head scenario for CIL
        seed=1234
    )

    # Model
    model = resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 100)

    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=config["lr"],
        momentum=0.9,
        weight_decay=5e-4
    )


    criterion = nn.CrossEntropyLoss()

    # Logger & Metrics
    interactive_logger = InteractiveLogger()
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(stream=True),
        forgetting_metrics(stream=True),
        loggers=[interactive_logger]
    )

    # Strategy
    strategy = EWC(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        ewc_lambda=config["ewc_lambda"],
        mode=config["mode"],
        decay_factor=config["decay_factor"],
        train_mb_size=config["batch_size"],
        train_epochs=config["train_epochs"],
        eval_mb_size=256,
        device=device,
        evaluator=eval_plugin
    )

    results = []

    for exp in benchmark.train_stream:
        print(f"\n--- Training on experience {exp.current_experience} ---")

        strategy.train(exp)
        strategy.eval(benchmark.test_stream)

        last_metrics = eval_plugin.get_last_metrics()

        stream_acc = last_metrics.get(
            "Top1_Acc_Stream/eval_phase/test_stream/Task000", 0.0
        )

        stream_forgetting = last_metrics.get(
            "StreamForgetting/eval_phase/test_stream", 0.0
        )

        results.append((stream_acc, stream_forgetting))

    # Compute final metrics
    all_acc = [r[0] for r in results]
    all_forg = [r[1] for r in results]

    AA = all_acc[-1]                           # Final Average Accuracy; “How good is the final model?”
    AIA = sum(all_acc) / len(all_acc)          # Average of StreamAccuracy over all experiences; “How good was the model during the whole learning process?”
    FM = all_forg[-1]                          # Forgetting Measure

    print("\n===== FINAL METRICS =====")
    print(f"AA  = {AA:.4f}")
    print(f"AIA = {AIA:.4f}")
    print(f"FM  = {FM:.4f}")

    return AA, AIA, FM


# ============================================================
# LOGGING FUNCTION
# ============================================================
def log_results(method_name, config_dict, AA, AIA, FM,
                filename="EWC_cifar100_log.csv"):

    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=[
            "date",
            "method",
            "ewc_lambda",
            "mode",
            "decay_factor",
            "lr",
            "train_epochs",
            "batch_size",
            "AA",
            "AIA",
            "FM"
        ])

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "method": method_name,
            "ewc_lambda": config_dict["ewc_lambda"],
            "mode": config_dict["mode"],
            "decay_factor": config_dict["decay_factor"],
            "lr": config_dict["lr"],
            "train_epochs": config_dict["train_epochs"],
            "batch_size": config_dict["batch_size"],
            "AA": round(AA, 4),
            "AIA": round(AIA, 4),
            "FM": round(FM, 4)
        })


# ============================================================
# MAIN LOOP (MULTI-CONFIG)
# ============================================================
def main():

    # --------------------------------------------------------
    # Base configuration (fixed parameters)
    # --------------------------------------------------------
    base_config = {
        "mode": "online", #mode="separate"
        "decay_factor": 0.95, #[0.9, 0.95, 0.99]
        "lr": 0.01,
        "train_epochs": 30,
        "batch_size": 128
    }

    # --------------------------------------------------------
    # Lambda sweep (controlled experiment)
    # --------------------------------------------------------
    lambdas = [0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 200.0, 500.0]

    configs = []
    for lam in lambdas:
        config = base_config.copy()
        config["ewc_lambda"] = lam
        configs.append(config)

    # --------------------------------------------------------
    # Run all configs
    # --------------------------------------------------------
    for i, config in enumerate(configs):

        print("\n==================================================")
        print(f"Running configuration {i+1}/{len(configs)}")
        print(config)
        print("==================================================")

        AA, AIA, FM = run_experiment(config)

        log_results("EWC", config, AA, AIA, FM)

    print("\n===== ALL EXPERIMENTS FINISHED =====")


if __name__ == "__main__":
    main()
