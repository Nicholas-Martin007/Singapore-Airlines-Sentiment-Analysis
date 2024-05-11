import torch
import torch.nn as nn
import torch.optim

import pandas as pd
import numpy as np

import time, sys, os

def test_sentiment_analysis(model, test_loader):
    
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Test in {device}")
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        model.eval()

        running_loss = 0
        running_error = 0
        correct = 0
        total = 0

        for i, data in enumerate(test_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)

            running_error += (predicted != labels).long().sum().item()
            correct += (predicted == labels).long().sum().item()
            total += labels.size(0)


        avg_test_error = running_error/len(test_loader.dataset)
        avg_test_loss = running_loss/len(test_loader)

        test_acc = correct/total

        t = time.time() - start_time
        print(f"Test_Loss={avg_test_loss:.4f}, Test_Error={avg_test_error:.4f}, Test_Acc={test_acc:.4%}")
    