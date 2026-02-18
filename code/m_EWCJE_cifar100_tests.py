# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 15:18:54 2026

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

import torch.nn.functional as F

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
# EWC-JE STRATEGY (Custom)
# ============================================================
class EWCJE(EWC):
    def __init__(self, *args, entropy_beta=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.entropy_beta = entropy_beta

    def criterion(self):
        # Standard cross entropy
        ce_loss = super().criterion()

        # Entropy term
        outputs = self.mb_output
        probs = F.softmax(outputs, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()

        # EWC + entropy maximization
        total_loss = ce_loss - self.entropy_beta * entropy

        return total_loss


# ============================================================
# SINGLE EXPERIMENT
# ============================================================
def run_experiment(config):

    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    benchmark = SplitCIFAR100(
        n_experiences=10,
        return_task_id=False,
        seed=1234
    )

    model = resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 100)

    optimizer = optim.SGD(
        model.parameters(),
        lr=config["lr"],
        momentum=0.9,
        weight_decay=5e-4
    )

    interactive_logger = InteractiveLogger()
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(stream=True),
        forgetting_metrics(stream=True),
        loggers=[interactive_logger]
    )

    strategy = EWCJE(
        model=model,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        ewc_lambda=config["ewc_lambda"],
        entropy_beta=config["entropy_beta"],
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

    all_acc = [r[0] for r in results]
    all_forg = [r[1] for r in results]

    AA = all_acc[-1]
    AIA = sum(all_acc) / len(all_acc)
    FM = all_forg[-1]

    print("\n===== FINAL METRICS =====")
    print(f"AA  = {AA:.4f}")
    print(f"AIA = {AIA:.4f}")
    print(f"FM  = {FM:.4f}")

    return AA, AIA, FM


# ============================================================
# LOGGING FUNCTION
# ============================================================
def log_results(method_name, config_dict, AA, AIA, FM,
                filename="EWCJE_cifar100_log.csv"):

    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=[
            "date",
            "method",
            "ewc_lambda",
            "entropy_beta",
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
            "entropy_beta": config_dict["entropy_beta"],
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
# MAIN
# ============================================================
def main():

    base_config = {
        "mode": "online",
        "decay_factor": 0.95,
        "lr": 0.01,
        "train_epochs": 30,
        "batch_size": 128
    }

    lambdas = [10, 50, 100, 200]
    entropys = [0.05, 0.1, 0.2]

    configs = []
    for lam in lambdas:
        for entropy in entropys:
            config = base_config.copy()
            config["ewc_lambda"] = lam
            config["entropy_beta"] = entropy
            configs.append(config)

    for i, config in enumerate(configs):

        print("\n==================================================")
        print(f"Running configuration {i+1}/{len(configs)}")
        print(config)
        print("==================================================")

        AA, AIA, FM = run_experiment(config)

        log_results("EWC-EJ", config, AA, AIA, FM)

    print("\n===== ALL EXPERIMENTS FINISHED =====")


if __name__ == "__main__":
    main()
