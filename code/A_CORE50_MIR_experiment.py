# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 11:56:46 2026

@author: mbinsnka280
"""
"""
# ============================================
# MIR — CORe50 NC Scenario (Class Incremental)
# ============================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from torchvision.models import resnet18

from avalanche.benchmarks.classic import CORe50
from avalanche.training.supervised import MIR
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger
import os
import pandas as pd

RESULTS_DIR = "mir_core50_results"
os.makedirs(RESULTS_DIR, exist_ok=True)
# ================= CONFIG =================
SEED = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


EPOCHS = 30
MEMORY_SIZE = 5000

torch.manual_seed(SEED)
np.random.seed(SEED)

print("Device:", DEVICE)

# ================= BENCHMARK =================
benchmark = CORe50(
    scenario="nc",        # Class Incremental
    run=0
)

NUM_CLASSES = benchmark.n_classes

# ================= MODEL =================
model = resnet18(num_classes=NUM_CLASSES)

# Adaptation small images
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()

model.to(DEVICE)

# ================= OPTIMIZER =================
optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

# ================= METRICS =================
logger = InteractiveLogger()

eval_plugin = EvaluationPlugin(
    accuracy_metrics(stream=True, experience=True),
    forgetting_metrics(stream=True, experience=True),
    loggers=[logger]
)

# ================= STRATEGY (MIR) =================
strategy = MIR(
    model,
    optimizer,
    criterion,
    mem_size=MEMORY_SIZE,
    subsample=512,
    train_mb_size=64,
    train_epochs=EPOCHS,
    eval_mb_size=128,
    device=DEVICE,
    evaluator=eval_plugin
)


# ================= TRAIN LOOP =================
stream_acc = []
stream_forg = []
accuracy_matrix = []

print("\n===== START TRAINING MIR — CORe50 =====")

for exp in benchmark.train_stream:

    print(f"\nTraining Experience {exp.current_experience}")

    strategy.train(exp)
    results = strategy.eval(benchmark.test_stream)

    # ---------------- STREAM ACCURACY ----------------
    acc_val = None
    for k, v in results.items():
        if "Top1_Acc_Stream" in k:
            acc_val = v
    stream_acc.append(acc_val if acc_val is not None else 0)

    # ---------------- STREAM FORGETTING ----------------
    forg_val = None
    for k, v in results.items():
        if "StreamForgetting/eval_phase/test_stream" in k:
            forg_val = v
    stream_forg.append(forg_val if forg_val is not None else 0)

    # ---------------- PER-TASK ACCURACY ----------------
    exp_acc = []
    for t in range(len(benchmark.test_stream)):
        key = f"Top1_Acc_Exp/eval_phase/test_stream/Exp{t:03d}"
        exp_acc.append(results.get(key, 0))
    accuracy_matrix.append(exp_acc)
# ================= FINAL METRICS =================
accuracy_matrix = np.array(accuracy_matrix)

final_acc = accuracy_matrix[-1]
best_acc = np.max(accuracy_matrix, axis=0)
forgetting_per_task = best_acc - final_acc

AA = np.mean(final_acc)
FM = np.mean(forgetting_per_task)

print("\n===== FINAL RESULTS =====")
print("Average Accuracy:", AA)
print("Mean Forgetting:", FM)

# ================= SAVE CSV =================

pd.DataFrame(accuracy_matrix).to_csv(
    os.path.join(RESULTS_DIR, "accuracy_matrix.csv"),
    index=False
)

pd.DataFrame({
    "Final Accuracy": final_acc,
    "Best Accuracy": best_acc,
    "Forgetting": forgetting_per_task
}).to_csv(
    os.path.join(RESULTS_DIR, "forgetting_per_task.csv"),
    index=False
)

pd.DataFrame({
    "Stream Accuracy": stream_acc,
    "Stream Forgetting": stream_forg
}).to_csv(
    os.path.join(RESULTS_DIR, "stream_metrics.csv"),
    index=False
)

# Save summary
with open(os.path.join(RESULTS_DIR, "summary.txt"), "w") as f:
    f.write(f"Average Accuracy: {AA}\n")
    f.write(f"Mean Forgetting: {FM}\n")


# ================= PLOTS =================
x = list(range(len(stream_acc)))

# Stream Accuracy
plt.figure()
plt.plot(x, stream_acc, marker='o')
plt.title("Stream Accuracy — MIR CORe50")
plt.xlabel("Experience")
plt.ylabel("Accuracy")
plt.grid()
plt.savefig(os.path.join(RESULTS_DIR, "stream_accuracy.png"))
plt.show()

# Stream Forgetting
plt.figure()
plt.plot(x, stream_forg, marker='o')
plt.title("Stream Forgetting — MIR CORe50")
plt.xlabel("Experience")
plt.ylabel("Forgetting")
plt.grid()
plt.savefig(os.path.join(RESULTS_DIR, "stream_forgetting.png"))
plt.show()

# Per-task accuracy
plt.figure()
for t in range(accuracy_matrix.shape[1]):
    plt.plot(accuracy_matrix[:, t], marker='o', label=f"Task {t}")
plt.title("Per-task Accuracy")
plt.xlabel("Experience")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.savefig(os.path.join(RESULTS_DIR, "per_task_accuracy.png"))
plt.show()

# Forgetting per task
plt.figure()
plt.bar(range(len(forgetting_per_task)), forgetting_per_task)
plt.title("Forgetting per Task")
plt.xlabel("Task")
plt.ylabel("Forgetting")
plt.grid()
plt.savefig(os.path.join(RESULTS_DIR, "forgetting_per_task.png"))
plt.show()
"""
# ============================================
# MIR — CORe50
# ============================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torchvision.models import resnet18
from avalanche.benchmarks.classic import CORe50
from avalanche.training.supervised import MIR
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger


# ================= CONFIG =================
SEED = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 10
LR = 0.01
SUBSAMPLE = 256

#MEMORY_SIZES = [500, 1500, 3000, 5000]  
MEMORY_SIZES = [3000] 

RESULTS_DIR = "mir_core50_memory_compare"
os.makedirs(RESULTS_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

print("Device:", DEVICE)


# ================= BENCHMARK =================
benchmark = CORe50(scenario="nc", run=0)
NUM_CLASSES = benchmark.n_classes


# ================= STORAGE =================
all_acc = {}
all_forg = {}


# ============================================
# LOOP OVER MEMORY SIZES
# ============================================
for MEM in MEMORY_SIZES:

    print("\n===================================")
    print(f"Running MIR with mem_size = {MEM}")
    print("===================================")

    # ---- Model reset ----
    model = resnet18(num_classes=NUM_CLASSES)
    model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = nn.Identity()
    model.to(DEVICE)

    optimizer = optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=0.9,
        weight_decay=5e-4
    )

    criterion = nn.CrossEntropyLoss()

    logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(stream=True),
        forgetting_metrics(stream=True),
        loggers=[logger]
    )

    strategy = MIR(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        mem_size=MEM,
        subsample=SUBSAMPLE,
        train_mb_size=64,
        train_epochs=EPOCHS,
        eval_mb_size=128,
        device=DEVICE,
        evaluator=eval_plugin
    )

    stream_acc = []
    stream_forg = []

    # -------- TRAIN LOOP --------
    for exp in benchmark.train_stream:

        print(f"Experience {exp.current_experience}")

        strategy.train(exp)
        results = strategy.eval(benchmark.test_stream)

        acc = results.get(
            "Top1_Acc_Stream/eval_phase/test_stream", 0.0
        )

        forg = results.get(
            "StreamForgetting/eval_phase/test_stream", 0.0
        )

        stream_acc.append(acc)
        stream_forg.append(forg)

    all_acc[MEM] = stream_acc
    all_forg[MEM] = stream_forg


# ============================================
# SAVE RESULTS
# ============================================
pd.DataFrame(all_acc).to_csv(
    os.path.join(RESULTS_DIR, "accuracy_all_memories.csv")
)

pd.DataFrame(all_forg).to_csv(
    os.path.join(RESULTS_DIR, "forgetting_all_memories.csv")
)


# ============================================
# PLOT — ACCURACY
# ============================================
plt.figure(figsize=(8,5))

for MEM in MEMORY_SIZES:
    plt.plot(all_acc[MEM], marker='o', label=f"Mem={MEM}")

plt.title("MIR — Accuracy vs Experience (CORe50)")
plt.xlabel("Experience")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "accuracy_multi_memory.png"))
plt.show()


# ============================================
# PLOT — FORGETTING
# ============================================
plt.figure(figsize=(8,5))

for MEM in MEMORY_SIZES:
    plt.plot(all_forg[MEM], marker='o', label=f"Mem={MEM}")

plt.title("MIR — Forgetting vs Experience (CORe50)")
plt.xlabel("Experience")
plt.ylabel("Forgetting")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "forgetting_multi_memory.png"))
plt.show()


