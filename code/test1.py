from avalanche.benchmarks.classic import SplitCIFAR10
from avalanche.models import SimpleCNN
from avalanche.training.supervised import Naive
import torch

def main():
    # Scenario: Class-Incremental Learning
    benchmark = SplitCIFAR10(
        n_experiences=5,
        seed=0
    )

    # Model
    model = SimpleCNN(num_classes=10)

    # Strategy: Naive (baseline)
    strategy = Naive(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        criterion=torch.nn.CrossEntropyLoss(),
        train_mb_size=32,
        eval_mb_size=32,
        device="cpu"
    )

    # Continual learning loop
    for exp in benchmark.train_stream:
        print(f"\n--- Training on experience {exp.current_experience} ---")
        strategy.train(exp)
        strategy.eval(benchmark.test_stream)

    print("\n✅ Baseline Naive finished")

if __name__ == "__main__":
    main()
