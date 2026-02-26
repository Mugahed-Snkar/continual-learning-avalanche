
# ============================================
# A-GEM — SPLIT CIFAR100
# ============================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.models import resnet18

from avalanche.benchmarks.classic import SplitCIFAR100
from avalanche.training.supervised import AGEM
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger

# ============================================
# CONFIG
# ============================================

SEED = 1
N_EXPERIENCES = 10
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_DIR = "agem_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================
# TRANSFORMS
# ============================================

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.507,0.487,0.441),
                         (0.267,0.256,0.276))
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.507,0.487,0.441),
                         (0.267,0.256,0.276))
])

# ============================================
# BENCHMARK — SINGLE HEAD
# ============================================

benchmark = SplitCIFAR100(
    n_experiences=N_EXPERIENCES,
    seed=SEED,
    train_transform=train_transform,
    eval_transform=eval_transform,
    class_ids_from_zero_in_each_exp=False
)

# ============================================
# MODEL — ResNet18 CIFAR
# ============================================

model = resnet18(num_classes=100)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
model.to(DEVICE)

# ============================================
# OPTIMIZER
# ============================================

optimizer = optim.SGD(
    model.parameters(),
    lr=0.05,
    momentum=0.9,
    weight_decay=5e-4
)

scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[20, 40],
    gamma=0.1
)

criterion = nn.CrossEntropyLoss()

# ============================================
# METRICS
# ============================================

logger = InteractiveLogger()

eval_plugin = EvaluationPlugin(
    accuracy_metrics(experience=True, stream=True),
    forgetting_metrics(experience=True),
    loggers=[logger]
)

# ============================================
# STRATEGY — A-GEM (Stable memory)
# ============================================

strategy = AGEM(
    model=model,
    optimizer=optimizer,
    criterion=criterion,

    patterns_per_exp=5000,     
    sample_size=256,

    train_mb_size=32,         
    eval_mb_size=64,
    train_epochs=EPOCHS,

    device=DEVICE,
    evaluator=eval_plugin
)

# ============================================
# STORAGE
# ============================================

accuracy_matrix = []
stream_acc = []

print("\nStarting A-GEM Training")

# ============================================
# TRAIN LOOP
# ============================================

for exp_id, experience in enumerate(benchmark.train_stream):

    print(f"\n===== Experience {exp_id} =====")

    strategy.train(experience)
    scheduler.step()

    results = strategy.eval(benchmark.test_stream)

    # ---------- STREAM ACC ----------
    stream_key = [k for k in results if "Top1_Acc_Stream" in k][0]
    stream_acc.append(results[stream_key])

    # ---------- PER TASK ACC ----------
    exp_acc = []
    for t in range(N_EXPERIENCES):
        key = f"Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp{t:03d}"
        exp_acc.append(results.get(key, 0.0))

    accuracy_matrix.append(exp_acc)

# ============================================
# METRICS COMPUTATION
# ============================================

accuracy_matrix = np.array(accuracy_matrix)

final_acc = accuracy_matrix[-1]
best_acc = np.max(accuracy_matrix, axis=0)
forgetting = best_acc - final_acc

AA = np.mean(final_acc)
FM = np.mean(forgetting)

print("\n===== FINAL RESULTS =====")
print("Average Accuracy (AA):", round(AA,4))
print("Mean Forgetting (FM):", round(FM,4))

# ============================================
# SAVE CSV
# ============================================

pd.DataFrame(accuracy_matrix).to_csv(
    os.path.join(RESULTS_DIR, "accuracy_matrix.csv"),
    index=False
)

pd.DataFrame({
    "Final Accuracy": final_acc,
    "Best Accuracy": best_acc,
    "Forgetting": forgetting
}).to_csv(
    os.path.join(RESULTS_DIR, "forgetting.csv"),
    index=False
)

# ============================================
# PLOTS
# ============================================

# Stream accuracy
plt.figure()
plt.plot(stream_acc, marker='o')
plt.title("Stream Accuracy (A-GEM CIFAR100)")
plt.xlabel("Experience")
plt.ylabel("Accuracy")
plt.grid()
plt.savefig(os.path.join(RESULTS_DIR, "stream_accuracy.png"))
plt.show()

# Per-task accuracy
plt.figure()
for t in range(N_EXPERIENCES):
    plt.plot(accuracy_matrix[:, t], marker='o', label=f"Task {t}")
plt.title("Per-task Accuracy")
plt.xlabel("Experience")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.savefig(os.path.join(RESULTS_DIR, "per_task_accuracy.png"))
plt.show()

# Forgetting
plt.figure()
plt.bar(range(N_EXPERIENCES), forgetting)
plt.title("Forgetting per Task")
plt.xlabel("Task")
plt.ylabel("Forgetting")
plt.grid()
plt.savefig(os.path.join(RESULTS_DIR, "forgetting.png"))
plt.show()
