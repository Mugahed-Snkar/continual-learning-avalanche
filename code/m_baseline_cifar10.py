import torch
import torch.nn as nn
import torch.optim as optim

from avalanche.benchmarks.classic import SplitCIFAR10

from torchvision.models import resnet18
from avalanche.models import SimpleCNN

from avalanche.training.supervised import Naive
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger


def main():
    
    # -----------------------------
    # Device
    # -----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")


    # ===============================
    # Benchmark (Class-Incremental)
    # ===============================
    benchmark = SplitCIFAR10(
        n_experiences=5,          # 2 classes per experience
        return_task_id=False,     # IMPORTANT for CIL
        seed=1234
    )


    # ===============================
    # Model
    # ===============================
    '''
    model = SimpleCNN(num_classes=10)
    '''
    model = resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.to(device)

    # -----------------------------
    # Optimizer & Loss
    # -----------------------------
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9
    )
    criterion = nn.CrossEntropyLoss()

    # ===============================
    # Logger & Metrics
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
    # Strategy: Naive
    # ===============================
    strategy = Naive(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_mb_size=32,
        train_epochs=1,
        eval_mb_size=32,
        device=device,
        evaluator=eval_plugin
    )
    

    # ===============================
    # Training & Evaluation Loop
    # ===============================
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


    # ===============================
    # Final Results Summary with AA, AIA, FM
    # ===============================
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
    AA = sum(all_acc) / len(all_acc)                  # Average Accuracy (final)
    AIA = sum([acc for acc in all_acc]) / len(results)  # Approximate AIA
    FM = sum(all_forg) / len(all_forg)               # Forgetting Measure
    
    print("\n===== ADDITIONAL METRICS =====")
    print(f"AA  = {AA:.4f}")
    print(f"AIA = {AIA:.4f}")
    print(f"FM  = {FM:.4f}")

    print("\n✅ Naive experiment finished")

if __name__ == "__main__":
    main()