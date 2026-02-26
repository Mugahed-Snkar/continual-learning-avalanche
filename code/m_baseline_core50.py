
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18

from avalanche.benchmarks.classic import CORe50
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
                     filename="Naive_core50_curve2.csv"):

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
    benchmark = CORe50(scenario="nc", run=0)
    print("Number of experiences:", len(benchmark.train_stream))

    # --------------------------------------------------------
    # Model (50 classes)
    # --------------------------------------------------------
    model = resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 50)
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
    print("\n===== NAIVE BASELINE EXPERIMENT =====")
    results = []

    for exp in benchmark.train_stream:
        print(f"\n--- Training on experience {exp.current_experience} ---")

        strategy.train(exp)
        strategy.eval(benchmark.test_stream)

        last_metrics = eval_plugin.get_last_metrics()
        stream_acc = last_metrics.get(
            "Top1_Acc_Stream/eval_phase/test_stream/Task000", None
        )
        stream_forgetting = last_metrics.get(
            "StreamForgetting/eval_phase/test_stream", None
        )
        results.append(
            (exp.current_experience, stream_acc, stream_forgetting)
        )

    # -----------------------------
    # Final Results Summary with AA, AIA, FM
    # -----------------------------
    print("\n===== FINAL RESULTS (NAIVE) =====")
    print("Experience | Accuracy (Stream) | Forgetting (Stream)")
    print("-----------------------------------------------------")
    
    # Store accuracies and forgetting for summary
    all_acc = []
    all_forg = []

    for exp_id, acc, forg in results:
        acc_val = acc if acc is not None else 0.0
        forg_val = forg if forg is not None else 0.0

        all_acc.append(acc_val)
        all_forg.append(forg_val)
        acc_str = f"{acc_val:.4f}"
        forg_str = f"{forg_val:.4f}"
        print(f"{exp_id:^10} | {acc_str:^17} | {forg_str:^20}")
    
    # Compute AA, AIA, FM
    AA = all_acc[-1]                   # Average Accuracy (final)
    AIA = sum(all_acc) / len(all_acc)  # Approximate AIA
    FM = eval_plugin.get_last_metrics().get(
        "StreamForgetting/eval_phase/test_stream", 0.0
    )

    print("\n===== ADDITIONAL METRICS =====")
    print(f"AA  = {AA:.4f}")
    print(f"AIA = {AIA:.4f}")
    print(f"FM  = {FM:.4f}")

    print("\nNaive experiment finished")


if __name__ == "__main__":
    main()
