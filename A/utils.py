import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm.auto import tqdm
import timm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

def train_model(model, train_loader, val_loader, n_epochs, lr, device, criterion):
    train_losses = []
    val_losses = []
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        # Training
        model.train()
        running_loss, total = 0.0, 0

        # tqdm Progess Bar setup
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            #forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            #backprop
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            total += images.size(0)
        avg_train_loss = running_loss / total
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss, val_total = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                #only forward pass since no training
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_total += images.size(0)
            avg_val_loss = val_loss / val_total
            val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    return model, train_losses, val_losses

# Just Saves a plot of training and validation losses vs epochs
def training_plots(train_losses,val_losses,n_epochs,title):
    plt.figure(figsize=(10,6))
    plt.plot(range(1, n_epochs+1), train_losses, label='Training Loss', marker='o')
    plt.plot(range(1, n_epochs+1), val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training And Validation Losses over Epochs')
    plt.legend()
    plt.grid(True)
    save_path = "Results/"+title+".png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Training Loss plot obtained and saved at: {save_path}")

# Generates Confusion Matrix and Accuracy value for any model
def test_model(model, data_loader, device):
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            targets.extend(labels.cpu().numpy())
    acc = accuracy_score(targets, preds)
    cm = confusion_matrix(targets, preds)
    return acc , cm

# Saves the earlier generated confusion matrix in a clean way
def confusion_mat(cm, class_names,title):
    num = len(class_names)
    fig , ax = plt.subplots()
    ax.imshow(cm,cmap='binary')
    ax.set_xticks(np.arange(0, num), labels=class_names,  rotation=90)
    ax.set_yticks(np.arange(0, num), labels=class_names)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('Actual Test Labels')
    plt.suptitle('Confusion Matrix')
    for i in range(num):
        for j in range(num):
            ax.text(i, j, cm[j, i], ha="center", va="center", color="r")
    save_path = "Results/"+title+".png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Confusion Matrix obtained and saved at: {save_path}")