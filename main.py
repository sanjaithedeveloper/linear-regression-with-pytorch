import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from data.dataProcessor import processDataset
from test import testModel
from train import trainModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"Running on GPU... {device}")
else:
    print(f"Running on CPU... {device}")
processedTensor = processDataset()
features = processedTensor[:, :-1]
labels = processedTensor[:, -1]
dataset = TensorDataset(features, labels)
trainSize = int(0.8 * len(dataset))
testSize = len(dataset) - trainSize
trainDataset, testDataset = random_split(dataset, [trainSize, testSize])
trainLoader = DataLoader(trainDataset, batch_size=2, shuffle=True)
testLoader = DataLoader(testDataset, batch_size=2, shuffle=False)
model = trainModel(features, trainLoader, device)
testModel(testLoader, device, model)
