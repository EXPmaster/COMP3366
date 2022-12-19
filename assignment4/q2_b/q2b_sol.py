import os
import cv2
import pandas as pd
from tqdm import tqdm

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, TwoLocal, StatePreparation, RealAmplitudes
from qiskit.quantum_info.operators import Operator
from qiskit import BasicAer, execute, QuantumCircuit, Aer


class DigitsDataset(Dataset):

    def __init__(self, data_path, label_path, label_of_interest=None) -> None:
        self.file = pd.read_csv(
            label_path,
            sep=',',
            header=None,
            names=["name", "class"]
        )
        self.imgs, self.labels = self.get(self.file, data_path)
        self.loi = label_of_interest
    
    def get(self, f, data_path):
        imgs, labels = [], []
        for i, (name, class_id) in enumerate(zip(f['name'], f['class'])):
            # if i > 10: break
            d_path = os.path.join(data_path, name)
            img = cv2.imread(d_path, 0) / 255.0
            # img = cv2.resize(img, (32, 32)).ravel()
            # if all(train_imgs == 0): continue
            imgs.append(img)
            labels.append(int(class_id))
        return imgs, labels

    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.labels[idx]
        if self.loi is not None:
            if label == self.loi:
                label = 1
            else:
                label = 0
        return (
            torch.tensor(img, dtype=torch.float32).unsqueeze(0),
            torch.tensor(label, dtype=torch.long)
        )
    
    def __len__(self):
        return len(self.imgs)
        

class Net(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=5)
        self.conv2 = nn.Conv2d(2, 16, kernel_size=5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 2)  # 2-dimensional input to QNN
        self.qnn = TorchConnector(self.create_qnn(num_classes))  # Apply torch connector, weights chosen
        # uniformly at random from interval [-1,1].
        self.fc3 = nn.Linear(1, 2)  # 1-dimensional output from QNN
        self.logsoftmax = nn.LogSoftmax()
    
    def create_qnn(self, num_classes):
        feature_map = ZZFeatureMap(num_classes)
        ansatz = RealAmplitudes(num_classes, reps=1)
        qc = QuantumCircuit(num_classes)
        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz, inplace=True)

        # REMEMBER TO SET input_gradients=True FOR ENABLING HYBRID GRADIENT BACKPROP
        qnn = EstimatorQNN(
            circuit=qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            input_gradients=True,
        )
        return qnn

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.qnn(x)  # apply QNN
        x = self.fc3(x)
        return self.logsoftmax(x)


def train(epoch, model, train_loader, optimizer, loss_func):
    total_loss = []  # Store loss history
    model.train()  # Set model to training mode

    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)  # Initialize gradient
        output = model(data)  # Forward pass
        loss = loss_func(output, target)  # Calculate loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize weights
        total_loss.append(loss.item())  # Store loss
    loss = sum(total_loss) / len(total_loss)
    print("Training [{}]\tLoss: {:.4f}".format((epoch + 1), loss))


def evaluate(model, test_loader):
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        correct = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            if len(output.shape) == 1:
                output = output.reshape(1, *output.shape)

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        print(
            "Performance on test data:\n\tAccuracy: {:.2f}%".format(
                correct / len(test_loader) / test_loader.batch_size * 100
            )
        )


def main():
    for loi in range(10):
        trainset = DigitsDataset(train_data_path, train_label_path, loi)
        testset = DigitsDataset(test_data_path, test_label_path, loi)
        batch_size = 1
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
        print('data loaded.')
        model = Net(num_classes=2).to(device)
        print('Start training...')
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_func = nn.NLLLoss(weight=torch.tensor([0.15, 0.85], device=device))

        # Start training
        epochs = 40  # Set number of epochs
        for epoch in range(epochs):
            train(epoch, model, train_loader, optimizer, loss_func)
            evaluate(model, test_loader)
        torch.save(model, os.path.join(model_path, f'{loi}.pt'))


@torch.no_grad()
def test():
    testset = DigitsDataset(test_data_path, test_label_path)
    test_loader = DataLoader(testset, batch_size=1, shuffle=False)
    models = []
    for loi in range(10):
        model = torch.load(os.path.join(model_path, f'{loi}.pt'), map_location='cpu')
        model.eval()
        model.to(device)
        models.append(model)

    correct = 0
    for batch_idx, (data, target) in tqdm(enumerate(test_loader)):
        data, target = data.to(device), target.to(device)

        preds = []
        for model in models:
            output = model(data)
            if len(output.shape) == 1:
                output = output.reshape(1, *output.shape)
            pred = output.argmax()
            preds.append(pred)
        preds = torch.stack(preds)
        final_pred = torch.argmax(preds)
        correct += final_pred.eq(target.view_as(final_pred)).sum().item()

    print(
        "Performance on test data:\n\tAccuracy: {:.2f}%".format(
            correct / len(test_loader) * 100
        )
    )

if __name__ == '__main__':
    train_label_path = './Mnist-localization-master/MNIST/training_data.csv'
    train_data_path = './Mnist-localization-master/MNIST/MNIST_Converted_Training'
    test_label_path = './Mnist-localization-master/MNIST/test_data.csv'
    test_data_path = './Mnist-localization-master/MNIST/MNIST_Converted_Testing'
    model_path = './weights'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
    print(device)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    # main()
    test()