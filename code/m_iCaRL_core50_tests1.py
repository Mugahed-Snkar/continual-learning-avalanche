
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18

from avalanche.benchmarks.classic import CORe50
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

    # Benchmark
    benchmark = CORe50(scenario="nc", run=0)
    n_experiences = len(benchmark.train_stream)
    print("Number of training experiences:", len(benchmark.train_stream))

    # Model
    backbone = resnet18(pretrained=True)
    in_features = backbone.fc.in_features
    backbone.fc = nn.Identity()
    classifier = nn.Linear(in_features, 50)

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
    accuracy_matrix = np.zeros((n_experiences, n_experiences))
    stream_curve = []

    # -------------------------------
    # TRAINING LOOP
    # -------------------------------
    for train_idx, exp in enumerate(benchmark.train_stream):
    
        print(f"\n--- Training on experience {train_idx} ---")
        strategy.train(exp)
    
        # --------------------------------------------------
        # Single evaluation on full test stream
        # --------------------------------------------------
        for test_idx, test_exp in enumerate(benchmark.test_stream):
        
            strategy.eval(test_exp)
            last_metrics = eval_plugin.get_last_metrics()
        
            key = f"Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp{test_idx:03d}"
            acc = last_metrics.get(key, 0.0)
        
            accuracy_matrix[train_idx, test_idx] = acc
        
        strategy.eval(benchmark.test_stream)
        last_metrics = eval_plugin.get_last_metrics()
        
        stream_acc = last_metrics.get(
            "Top1_Acc_Stream/eval_phase/test_stream/Task000", 0.0
        )
        stream_curve.append(stream_acc)

    # -------------------------------
    # FINAL METRICS
    # -------------------------------
    AA = stream_curve[-1]
    AIA = sum(stream_curve) / len(stream_curve)
    FM = last_metrics.get(
        "StreamForgetting/eval_phase/test_stream", 0.0
    )

    print("\n===== FINAL METRICS =====")
    print(f"AA  = {AA:.4f}")
    print(f"AIA = {AIA:.4f}")
    print(f"FM  = {FM:.4f}")

    return AA, AIA, FM, stream_curve, accuracy_matrix


# ============================================================
# LOGGING FUNCTION
# ============================================================
def log_results(method_name, config_dict, AA, AIA, FM,
                filename="ICaRL_core50_log.csv"):

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
                     filename="ICaRL_core50_curve.csv"):

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

def log_accuracy_matrix(matrix, config,
                        filename="ICaRL_core50_matrix.csv"):

    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            header = ["date", "memory_size",
                      "train_exp", "test_exp", "accuracy"]
            writer.writerow(header)

        n = matrix.shape[0]

        for i in range(n):
            for j in range(n):
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M"),
                    config["memory_size"],
                    i,
                    j,
                    round(matrix[i, j], 4)
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

        AA, AIA, FM, stream_curve, accuracy_matrix = run_experiment(config)
        
        log_results("ICaRL", config, AA, AIA, FM)
        log_stream_curve(stream_curve, config)
        log_accuracy_matrix(accuracy_matrix, config)


    print("\n===== ALL EXPERIMENTS FINISHED =====")


if __name__ == "__main__":
    main()
