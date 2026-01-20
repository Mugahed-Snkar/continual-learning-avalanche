# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 11:08:51 2026

@author: mbinsnka280
"""

from avalanche.benchmarks.classic import SplitCIFAR10
from avalanche.models import SimpleCNN
from avalanche.training.supervised import Naive
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.logging import InteractiveLogger
import torch



benchmark = SplitCIFAR10(n_experiences=2, seed=0)

model = SimpleCNN(num_classes=10)

logger = InteractiveLogger()

eval_plugin = EvaluationPlugin(
    accuracy_metrics(epoch=True, stream=True),
    loggers=[logger]
)

strategy = Naive(
    model,
    optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
    criterion=torch.nn.CrossEntropyLoss(),
    train_mb_size=32,
    eval_mb_size=32,
    device="cpu",
    evaluator=eval_plugin
)


for experience in benchmark.train_stream:
    print(f"\n--- Training on experience {experience.current_experience} ---")
    strategy.train(experience)
    strategy.eval(benchmark.test_stream)

print("Avalanche test terminé")