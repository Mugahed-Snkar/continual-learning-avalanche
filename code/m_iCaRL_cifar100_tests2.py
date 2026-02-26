# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 13:36:07 2026

@author: maboorai279
"""


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
    n_experiences = len(benchmark.train_stream)
    print("Number of training experiences:", n_experiences)

    # --------------------------------------------------------
    # Backbone (CIFAR-adapted ResNet18)
    # --------------------------------------------------------
    backbone = resnet18(pretrained=True)

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
        accuracy_metrics(stream=True, experience=True),
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

    # -------------------------------
    # STORAGE
    # -------------------------------
    stream_curve = []
    
    for train_idx, exp in enumerate(benchmark.train_stream):
        print(f"\n--- Training on experience {train_idx} ---")
        strategy.train(exp)
        
        # Evaluate on the full stream
        strategy.eval(benchmark.test_stream)
        
        # Get all metrics generated so far
        all_results = eval_plugin.get_all_metrics()
        
        # 1. Get Accuracies for ONLY seen tasks (Exp000 to Exp{train_idx})
        current_accuracies = []
        
        for i in range(train_idx + 1):
            key = f"Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp{i:03d}"
            # Get the latest value for this specific experience
            val = all_results[key][1][-1] 
            current_accuracies.append(val)
        
        # 2. Average of seen tasks
        current_seen_avg = sum(current_accuracies) / len(current_accuracies)
        stream_curve.append(current_seen_avg)
    
    # --------------------------------------------------------
    # FINAL METRICS
    # --------------------------------------------------------
    # AA: Accuracy of all 10 tasks at the very end
    AA = stream_curve[-1] 
    
    # AIA: Average of the seen-tasks-averages over time
    AIA = sum(stream_curve) / len(stream_curve)
    
    # FM: Average forgetting across the stream
    FM = eval_plugin.get_last_metrics().get(
        "StreamForgetting/eval_phase/test_stream", 0.0
    )
        
    print("\n===== FINAL METRICS =====")
    print(f"AA  = {AA:.4f}")
    print(f"AIA = {AIA:.4f}")
    print(f"FM  = {FM:.4f}")

    return AA, AIA, FM, stream_curve


# ============================================================
# LOGGING FUNCTION
# ============================================================
def log_results(method_name, config_dict, AA, AIA, FM,
                filename="ICaRL_cifar100_log2.csv"):

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

def log_stream_curve(stream_curve, config,
                     filename="ICaRL_cifar100_curve2.csv"):

    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            header = ["date", "memory_size", "experience", "stream_accuracy"]
            writer.writerow(header)

        for exp_id, acc in enumerate(stream_curve):
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M"),
                config["memory_size"],
                exp_id,
                round(acc, 4)
            ])


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

        AA, AIA, FM, stream_curve = run_experiment(config)
        
        log_results("ICaRL", config, AA, AIA, FM)
        log_stream_curve(stream_curve, config)


    print("\n===== ALL EXPERIMENTS FINISHED =====")


if __name__ == "__main__":
    main()
