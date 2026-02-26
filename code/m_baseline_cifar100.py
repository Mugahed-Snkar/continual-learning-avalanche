
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18

from avalanche.benchmarks.classic import SplitCIFAR100
from avalanche.training.supervised import Naive
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger

import random
import numpy as np

import csv
import os
from datetime import datetime


def log_stream_curve(stream_curve,
                     filename="Naive_cifar100_curve2.csv"):

    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            header = ["date", "method", "experience", "stream_accuracy"]
            writer.writerow(header)

        for exp_id, acc in enumerate(stream_curve):
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Naive",
                exp_id,
                round(acc, 4)
            ])


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


def main():

    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------
    benchmark = SplitCIFAR100(
        n_experiences=10,         # 2 classes per experience
        return_task_id=False,     # Single-head scenario for CIL
        seed=1234
    )
    print("Number of experiences:", len(benchmark.train_stream))

    # --------------------------------------------------------
    # Model (50 classes!)
    # --------------------------------------------------------
    model = resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 100)
    model.to(device)

    # --------------------------------------------------------
    # Optimizer & Loss
    # --------------------------------------------------------
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4
    )

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Evaluation Plugin
    # --------------------------------------------------------
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(stream=True, experience=True),
        forgetting_metrics(stream=True),
        loggers=[interactive_logger]
    )

    # --------------------------------------------------------
    # Naive Strategy
    # --------------------------------------------------------
    strategy = Naive(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_mb_size=128,
        train_epochs=50,
        eval_mb_size=256,
        device=device,
        evaluator=eval_plugin
    )

    # --------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------
    stream_curve = []

    print("\n===== NAIVE BASELINE =====")

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

    
    log_stream_curve(stream_curve)
    
    print("\nNaive experiment finished.")


if __name__ == "__main__":
    main()
