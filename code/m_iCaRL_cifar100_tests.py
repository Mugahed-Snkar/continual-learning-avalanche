# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18

from avalanche.benchmarks.classic import SplitCIFAR100
from avalanche.training.supervised import ICaRL
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import LRSchedulerPlugin

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

    # --------------------------------------------------------
    # Benchmark (Class-Incremental CIFAR100)
    # --------------------------------------------------------
    benchmark = SplitCIFAR100(
        n_experiences=10,      # 10 experiences × 10 classes
        return_task_id=False,  # Single-head (Class-IL)
        seed=1234
    )

    # --------------------------------------------------------
    # Backbone (CIFAR-adapted ResNet18)
    # --------------------------------------------------------
    backbone = resnet18(pretrained=False)

    # Adapt for 32x32 images
    backbone.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    backbone.maxpool = nn.Identity()

    num_features = backbone.fc.in_features
    backbone.fc = nn.Identity()  # remove classifier

    # Classifier for ALL classes (100)
    classifier = nn.Linear(num_features, 100)

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------
    optimizer = optim.SGD(
        list(backbone.parameters()) + list(classifier.parameters()),
        lr=config["lr"],
        momentum=0.9,
        weight_decay=5e-4
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[30, 40],
        gamma=0.1
    )
    scheduler_plugin = LRSchedulerPlugin(scheduler)


    # --------------------------------------------------------
    # Logger & Metrics
    # --------------------------------------------------------
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(stream=True),
        forgetting_metrics(stream=True),
        loggers=[interactive_logger]
    )

    # --------------------------------------------------------
    # Strategy (ICaRL)
    # --------------------------------------------------------
    strategy = ICaRL(
        feature_extractor=backbone,
        classifier=classifier,
        optimizer=optimizer,
        plugins=[scheduler_plugin],
        memory_size=config["memory_size"],
        buffer_transform=None,
        fixed_memory=True,
        train_mb_size=config["batch_size"],
        train_epochs=config["train_epochs"],
        eval_mb_size=256,
        device=device,
        evaluator=eval_plugin
    )

    # --------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Compute Metrics
    # --------------------------------------------------------
    all_acc = [r[0] for r in results]
    all_forg = [r[1] for r in results]

    AA = all_acc[-1]                    # Final Stream Accuracy
    AIA = sum(all_acc) / len(all_acc)   # Average over time
    FM = all_forg[-1]                   # Final Forgetting

    print("\n===== FINAL METRICS =====")
    print(f"AA  = {AA:.4f}")
    print(f"AIA = {AIA:.4f}")
    print(f"FM  = {FM:.4f}")

    return AA, AIA, FM


# ============================================================
# LOGGING FUNCTION
# ============================================================
def log_results(method_name, config_dict, AA, AIA, FM,
                filename="ICaRL_cifar100_log.csv"):

    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=[
            "date",
            "method",
            "memory_size",
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
            "memory_size": config_dict["memory_size"],
            "lr": config_dict["lr"],
            "train_epochs": config_dict["train_epochs"],
            "batch_size": config_dict["batch_size"],
            "AA": round(AA, 4),
            "AIA": round(AIA, 4),
            "FM": round(FM, 4)
        })


# ============================================================
# MAIN LOOP (Memory Sweep)
# ============================================================
def main():

    base_config = {
        "lr": 0.1,
        "train_epochs": 50,
        "batch_size": 128
    }

    # Sweep MEMORY size (important for ICaRL)
    memory_sizes = [500, 1000, 2000, 5000]

    configs = []
    for mem in memory_sizes:
        config = base_config.copy()
        config["memory_size"] = mem
        configs.append(config)

    for i, config in enumerate(configs):

        print("\n==================================================")
        print(f"Running configuration {i+1}/{len(configs)}")
        print(config)
        print("==================================================")

        AA, AIA, FM = run_experiment(config)

        log_results("ICaRL", config, AA, AIA, FM)

    print("\n===== ALL EXPERIMENTS FINISHED =====")


if __name__ == "__main__":
    main()
