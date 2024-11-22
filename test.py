import os
import numpy as np
import torch
from config import G_MODEL_PATH
from train import SimpleModel


def testModel(testLoader, device, model):
    modelPath = os.path.join(G_MODEL_PATH,'simpleModel.pth')
    model.load_state_dict(torch.load(modelPath))
    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        for inputs, labels in testLoader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = outputs.squeeze()
            total += labels.size(0)
            correct += (predicted.round() == labels).sum().item()
        print(f"Accuracy: {100 * correct / total:.2f}%")