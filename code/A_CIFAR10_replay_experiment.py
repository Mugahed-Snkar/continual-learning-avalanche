# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 11:22:03 2026

@author: mbinsnka280
"""


import torch
from avalanche.benchmarks.classic import SplitCIFAR10
from avalanche.models import SimpleCNN
from avalanche.training.supervised import Replay
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger


# ===============================
# Device
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def main():
    # ===============================
    # 1. Benchmark (Class-Incremental)
    # ===============================
    benchmark = SplitCIFAR10(
        n_experiences=5,
        seed=0
    )

    # ===============================
    # 2. Model
    # ===============================
    model = SimpleCNN(num_classes=10)
    

    # ===============================
    # 3. Logger & Metrics
    # ===============================
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            experience=True,
            stream=True
        ),
        forgetting_metrics(
            experience=True,
            stream=True
        ),
        loggers=[interactive_logger]
    )

    # ===============================
    # 4. Strategy: Replay
    # ===============================
    strategy = Replay(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.03,momentum=0.9),
        criterion=torch.nn.CrossEntropyLoss(),
        train_mb_size=128,
        eval_mb_size=128,
        train_epochs = 5,
        device=device,
        evaluator=eval_plugin,
        mem_size=500)
    
    
    
    # 4. Strategy: Naive


    # ==============================
    # 5. Training & Evaluation Loop
    # ==============================
    print("\n===== REPLAY EXPERIMENT =====")

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

    # ===============================
    # 6. Final Results Summary
    # ===============================
    print("\n===== FINAL RESULTS (REPLAY) =====")
    print("Experience | Accuracy (Stream) | Forgetting (Stream)")
    print("-----------------------------------------------------")

    for exp_id, acc, forg in results:
        acc_str = f"{acc:.4f}" if acc is not None else "N/A"
        forg_str = f"{forg:.4f}" if forg is not None else "N/A"
        print(f"{exp_id:^10} | {acc_str:^17} | {forg_str:^20}")

    print("\n✅ Replay experiment finished")


if __name__ == "__main__":
    main()
