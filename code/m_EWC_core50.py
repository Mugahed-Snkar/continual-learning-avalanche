import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18

from avalanche.benchmarks.classic import CORe50
from avalanche.training.supervised import EWC
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, forgetting_metrics, loss_metrics
from avalanche.logging import InteractiveLogger

import matplotlib.pyplot as plt

import csv
import os
from datetime import datetime

def main():
    
    # -----------------------------
    # Device
    # -----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # -----------------------------
    # Benchmark (Class-Incremental)
    # -----------------------------
    benchmark = CORe50(
        scenario="nc",   # or "ni", "nic"
        run=0
    )

    # -----------------------------
    # Model
    # -----------------------------
    model = resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 50)

    # -----------------------------
    # Optimizer & Loss
    # -----------------------------
    '''SGD is preferred for regularization-based methods (EWC)'''
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.05,
        momentum=0.9
    )
    criterion = nn.CrossEntropyLoss()

    # -----------------------------
    # Logger & Metrics
    # -----------------------------
    interactive_logger = InteractiveLogger()
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        loss_metrics(experience=True),
        loggers=[interactive_logger]
    )

    # -----------------------------
    # Strategy: EWC
    # -----------------------------  
    strategy = EWC(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        ewc_lambda=100.0,          
        mode="online",
        decay_factor=0.9,
        train_mb_size=128,
        train_epochs=50,
        eval_mb_size=256,
        device=device,
        evaluator=eval_plugin
    )


    print("\n===== MODEL PARAMETERS =====")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n===== EWC HYPERPARAMETERS =====")
    print(f"Number of experiences: {benchmark.n_experiences}")
    print(f"Train epochs: {strategy.train_epochs}")
    print(f"Train batch size: {strategy.train_mb_size}")


    # -----------------------------
    # Training & Evaluation Loop
    # -----------------------------
    print("\n===== EWC EXPERIMENT =====")
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
        train_loss = last_metrics.get(
            "Loss_Exp/train_phase/train_stream/Task000/Exp{:03d}".format(exp.current_experience), None
        )
        exp_acc = last_metrics.get(
            "Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp{:03d}".format(exp.current_experience), None
        )

        results.append(
            (exp.current_experience, stream_acc, exp_acc, stream_forgetting, train_loss)
        )

    # -----------------------------
    # Final Results Summary with AA, AIA, FM
    # -----------------------------
    print("\n===== FINAL RESULTS (EWC) =====")
    print("Experience | Accuracy (Stream) | Forgetting (Stream) | Accuracy (Experience)")
    print("----------------------------------------------------------------------------")

    # Store accuracies and forgetting for summary
    all_acc = []
    all_forg = []
    all_exp_acc = []
    
    for exp_id, stream_acc, exp_acc, forg, train_loss in results:

        acc_val = stream_acc if stream_acc is not None else 0.0
        forg_val = forg if forg is not None else 0.0
        exp_acc_val = exp_acc if exp_acc is not None else 0.0
        
        all_acc.append(acc_val)
        all_forg.append(forg_val)
        all_exp_acc.append(exp_acc_val)
        acc_str = f"{acc_val:.4f}"
        forg_str = f"{forg_val:.4f}"        
        exp_acc_str = f"{exp_acc_val:.4f}"
        
        print(f"{exp_id:^10} | {acc_str:^17} | {forg_str:^20} | {exp_acc_str:^23}")

    # Compute AA, AIA, FM
    AA = all_acc[-1]                   # Average Accuracy (final)
    AIA = sum(all_acc) / len(all_acc)  # Approximate AIA
    FM = sum(all_forg) / len(all_forg)            # Forgetting Measure

    print("\n===== ADDITIONAL METRICS =====")
    print(f"AA  = {AA:.4f}")
    print(f"AIA = {AIA:.4f}")
    print(f"FM  = {FM:.4f}")

    # Extract x and y values
    experiences = [r[0] for r in results]
    stream_accuracies = [r[1] if r[1] is not None else 0.0 for r in results]
    exp_accuracies = [r[2] if r[2] is not None else 0.0 for r in results]
    train_losses = [r[4] if r[4] is not None else 0.0 for r in results]

    # Plot
    plt.figure()
    plt.plot(experiences, stream_accuracies, marker='o')
    plt.xlabel("Experience")
    plt.ylabel("Accuracy (Stream)")
    plt.title("Accuracy (Stream) over Experiences")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(experiences, train_losses, marker='o')
    plt.xlabel("Experience")
    plt.ylabel("Training Loss")
    plt.title("Training Loss per Experience (EWC, CIFAR-100, CIL)")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(experiences, exp_accuracies, marker='o')
    plt.xlabel("Experience")
    plt.ylabel("Accuracy on Current Experience")
    plt.title("Per-Experience Accuracy (EWC, CIFAR-100, CIL)")
    plt.grid(True)
    plt.show()

    print("\n===== EWC experiment finished =====")

    config = {
        "ewc_lambda": 100.0,
        "mode": "online",
        "decay_factor": 0.9,
        "optimizer": "SGD",
        "lr": 0.05,
        "train_epochs": 50,
        "batch_size": 128
    }
    log_results("EWC", config, AA, AIA, FM)


def log_results(method_name, config_dict, AA, AIA, FM, filename="results_log.csv"):
    
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=[
            "date",
            "method",
            "ewc_lambda",
            "mode",
            "decay_factor",
            "optimizer",
            "lr",
            "train_epochs",
            "batch_size",
            "AA",
            "AIA",
            "FM"
        ])

        if not file_exists:
            writer.writeheader()

        row = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "method": method_name,
            "ewc_lambda": config_dict["ewc_lambda"],
            "mode": config_dict["mode"],
            "decay_factor": config_dict["decay_factor"],
            "optimizer": config_dict["optimizer"],
            "lr": config_dict["lr"],
            "train_epochs": config_dict["train_epochs"],
            "batch_size": config_dict["batch_size"],
            "AA": AA,
            "AIA": AIA,
            "FM": FM
        }

        writer.writerow(row)


if __name__ == "__main__":
    main()
