import numpy as np
import time
import copy
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
import torch.nn as nn

def train_model(model, optimizer, dataloaders, dataset_sizes,
                criterion = None, scheduler = None, 
                num_epochs = 1, patience = 10, 
                output_filename = "best_model.pth",
                use_gpu = False, device = "cpu",
                use_features = False):    
    since = time.time()
    
    # Initialize best metrics tracking
    best_metrics = {
        'epoch': 0,
        'val_loss': float('inf'),
        'val_accuracy': 0,
        'val_precision': 0,
        'val_recall': 0,
        'val_f1': 0,
    }
    
    best_model_wts = copy.deepcopy(model.state_dict())
    
    if not criterion:
        criterion = nn.CrossEntropyLoss()
    
    # use patience for early stopping when validation isnt getting better
    patience_left = patience

    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            epoch_start_time = time.time()
            if phase == "train":
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            all_preds = []
            all_labels = []
            
            for data in dataloaders[phase]:
                if use_features:
                    image, labels, features = data
                else:
                    image, labels = data
                if use_gpu:
                    labels = labels.to(device)
                
                optimizer.zero_grad()
                
                # forward
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(image, features) if use_features else model(image)
                    preds = torch.argmax(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * len(image)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_accuracy = accuracy_score(all_labels, all_preds)
            epoch_precision = precision_score(all_labels, all_preds, zero_division=0, average='macro')
            epoch_recall = recall_score(all_labels, all_preds, zero_division=0, average='macro')
            epoch_f1 = f1_score(all_labels, all_preds, zero_division=0, average='macro')

            epoch_time = time.time() - epoch_start_time

            # deep copy the model if it's best so far
            if phase == "val" and (epoch_loss < best_metrics['val_loss'] or epoch_recall > best_metrics['val_recall']):
                best_metrics.update({
                    'epoch': epoch + 1,
                    'val_loss': epoch_loss,
                    'val_accuracy': epoch_accuracy,
                    'val_precision': epoch_precision,
                    'val_recall': epoch_recall,
                    'val_f1': epoch_f1,
                })
                
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), output_filename)
                patience_left = patience
            elif phase == "val":
                patience_left -= 1
            
            print(f'Epoch {epoch}/{num_epochs - 1} {phase} complete in {epoch_time:.4f} seconds. {phase} loss: {epoch_loss:.4f} recall: {epoch_recall:.4f}. Patience left: {patience_left}')
            
        if patience_left <= 0:
            print("Ran out of patience. Stopping early")
            break
        
        if scheduler:
            scheduler.step()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f"Best Metrics at Epoch {best_metrics['epoch']}:")
    for metric, value in best_metrics.items():
        if metric != 'epoch':
            print(f"{metric.capitalize()}: {value:.4f}")

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model