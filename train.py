import torch
import torch.nn as nn
import torch.optim

import pandas as pd
import numpy as np

import time, sys, os

torch.manual_seed(1000)
np.random.seed(1000)

def model_name(name, batch_size, lr, epoch):
    path = f'model_{name}_bs_{batch_size}_lr_{lr}_epoch_{epoch}'
    return path

def train_sentiment_analysis(model, train_loader, val_loader, batch_size, lr, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Optimization => Adam
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=0.001)

    # Loss Function
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)

    print(f"Training in {device}")

    start_time = time.time()


    for epoch in range(epochs):
        running_loss = 0
        running_error = 0
        total = 0
        correct = 0

        model.train()

        for i, data in enumerate(train_loader):
            inputs, labels = data
            # ==========NOT USED===========
            # Changed because labels before substract is 1,2,3,4,5 which is not suitable
            # it should be 0,1,2,3,4 which contains 5 classes
            # labels -= 1
            # =============================
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            running_error += (predicted != labels).long().sum().item()
            correct += (predicted == labels).long().sum().item()
            total += labels.size(0)

        avg_train_error = running_error/len(train_loader.dataset)
        avg_train_loss = running_loss/len(train_loader)
        train_acc = correct/total

        model.eval()

        with torch.no_grad():
            running_loss = 0
            running_error = 0
            correct = 0
            total = 0

            for i, data in enumerate(val_loader):
                inputs, labels = data
                # ==========NOT USED===========
                # Changed because labels before substract is 1,2,3,4,5 which is not suitable
                # it should be 0,1,2,3,4 which contains 5 classes
                # labels -= 1
                # =============================
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)

                running_error += (predicted != labels).long().sum().item()
                correct += (predicted == labels).long().sum().item()
                total += labels.size(0)

            avg_val_error = running_error/len(val_loader.dataset)
            avg_val_loss = running_loss/len(val_loader)
            val_acc = correct/total

        t = time.time() - start_time
        print(f"Epoch {epoch+1}: Train_loss={avg_train_loss:.4f}, Train_Error={avg_train_error:.4f}, Train_Acc={train_acc:.4%} || Val_Loss = {avg_val_loss:.4f}, Val_Error={avg_val_error:.4f}, Val_Acc={val_acc:.4%}")
        
        if epoch+1 == 12:
            model_path = model_name(model.name, batch_size, lr, epoch+1)
            torch.save(model.state_dict(), model_path)
