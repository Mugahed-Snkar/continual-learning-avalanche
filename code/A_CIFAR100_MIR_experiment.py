# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 09:36:50 2026

@author: mbinsnka280
"""

import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import numpy as np

from torchvision import transforms
from avalanche.benchmarks.classic import SplitCIFAR100
from avalanche.training.supervised import MIR
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger


# ================= DEVICE =================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ================= MEMORY SIZES =================
MEMORY_SIZES = [500, 1000, 2000,3000]

# ================= DATA =================
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.507, 0.487, 0.441),
                         (0.267, 0.256, 0.276))
])

benchmark = SplitCIFAR100(
    n_experiences=10,
    seed=0,
    train_transform=train_transform
)

# ================= STORAGE =================
all_acc_curves = {}
all_forg_curves = {}

# ==========================================
# LOOP OVER MEMORY SIZES
# ==========================================
for MEM_SIZE in MEMORY_SIZES:

    print("\n==============================")
    print(f"Memory Size = {MEM_SIZE}")
    print("==============================")

    # ---- Model ----
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 100)
    model.to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,            # stable
        momentum=0.9,
        weight_decay=5e-4
    )

    logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(stream=True),
        forgetting_metrics(stream=True),
        loggers=[logger]
    )

    strategy = MIR(
        model=model,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        mem_size=MEM_SIZE,
        subsample=256,
        train_mb_size=128,
        eval_mb_size=128,
        train_epochs=50,
        device=device,
        evaluator=eval_plugin
    )

    acc_curve = []
    forg_curve = []

    # -------- TRAIN LOOP --------
    for exp in benchmark.train_stream:

        print(f"Experience {exp.current_experience}")

        strategy.train(exp)
        results = strategy.eval(benchmark.test_stream)

        # ----- Stream Accuracy -----
        acc = 0
        for k, v in results.items():
            if "Top1_Acc_Stream" in k:
                acc = v
        acc_curve.append(acc)

        # ----- Stream Forgetting -----
        forg = 0
        for k, v in results.items():
            if "StreamForgetting" in k:
                forg = v
        forg_curve.append(forg)

    all_acc_curves[MEM_SIZE] = acc_curve
    all_forg_curves[MEM_SIZE] = forg_curve


# ==========================================
# PLOT ACCURACY (ALL MEMORY ON SAME FIGURE)
# ==========================================
plt.figure(figsize=(8,5))

for mem in MEMORY_SIZES:
    plt.plot(all_acc_curves[mem], marker='o', label=f"Mem={mem}")

plt.title("MIR — Accuracy vs Experience (CIFAR100)")
plt.xlabel("Experience")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ==========================================
# PLOT FORGETTING (ALL MEMORY ON SAME FIGURE)
# ==========================================
plt.figure(figsize=(8,5))

for mem in MEMORY_SIZES:
    plt.plot(all_forg_curves[mem], marker='o', label=f"Mem={mem}")

plt.title("MIR — Forgetting vs Experience (CIFAR100)")
plt.xlabel("Experience")
plt.ylabel("Forgetting")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
