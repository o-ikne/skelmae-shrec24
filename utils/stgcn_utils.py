import torch
from einops import rearrange, repeat

from tqdm import tqdm
import os
import os.path as opt

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def train_one_epoch(epoch, num_epochs, model, mae, optimizer, dataloader, criterion, scheduler, device):
    model.train()
    
    pbar = tqdm(dataloader, total=len(dataloader))
    train_loss = 0

    for d in pbar:
        sequence = d['Sequence'].to(device)
        adj_mat = d['AM'].to(device)
        label = d['Label'].to(device)
        
        tokens = mae.inference(sequence, adj_mat)

        tokens = rearrange(tokens, 'b t n d -> b d t n')
        tokens = tokens.unsqueeze(-1)
        
        optimizer.zero_grad()
        pred = model(tokens)
        loss_train = criterion(pred, label)
        loss_train.backward()
        optimizer.step()
        
        train_loss += loss_train.item()
        pbar.set_description(f'[%.3g/%.3g] train loss. %.2f' % (epoch, num_epochs, train_loss))
        
        break


    if scheduler is not None:
        scheduler.step()
    
    return train_loss
        

def valid_one_epoch(model, mae, dataloader, criterion, device):
    accuracy = 0.0
    n = 0
    pred_labels, true_labels = [], []
    valid_loss = 0.0
    model.eval()
    with torch.no_grad():
        pbar = tqdm(dataloader, total=len(dataloader))
        for d in pbar:
            sequence = d['Sequence'].to(device)
            adj_mat = d['AM'].to(device)
            label = d['Label'].to(device)

            tokens = mae.inference(sequence, adj_mat)
            tokens = rearrange(tokens, 'b t n d -> b d t n')
            tokens = tokens.unsqueeze(-1)
        
            true_labels.extend(label.tolist())
            output = model(tokens)
            loss_valid = criterion(output, label)
            accuracy += (output.argmax(dim=1) == label.flatten()).sum().item()
            n += len(label.flatten())
            valid_loss += loss_valid.item()
            
            pred_labels.extend(output.argmax(dim=1).tolist())
            desc = '[VALID]> loss. %.2f > acc. %.2f%%' % (valid_loss, (accuracy / n)*100)
            pbar.set_description(desc)
            
            break


    accuracy = (accuracy / n)*100
                
    return valid_loss, accuracy, true_labels, pred_labels


def eval_stgcn(stgcn, mae, dataloader, device, args):

    save_folder_path = opt.join(args.save_folder_path, args.exp_name,'confusion_matrix/')
    os.makedirs(save_folder_path, exist_ok=True)

    accuracy = 0.0
    n = 0
    pred_labels, true_labels = [], []
    stgcn.eval()
    with torch.no_grad():
        pbar = tqdm(dataloader, total=len(dataloader))
        for d in pbar:
            sequence = d['Sequence'].to(device)
            adj_mat = d['AM'].to(device)
            label = d['Label'].to(device)

            tokens = mae.inference(sequence, adj_mat)
            tokens = rearrange(tokens, 'b t n d -> b d t n')
            tokens = tokens.unsqueeze(-1)
        
            true_labels.extend(label.tolist())
            output = stgcn(tokens)

            accuracy += (output.argmax(dim=1) == label.flatten()).sum().item()
            n += len(label.flatten())
            
            pred_labels.extend(output.argmax(dim=1).tolist())
            desc = '[VALID]> acc. %.2f%%' % ((accuracy / n)*100)
            pbar.set_description(desc)
            
            break

    accuracy = (accuracy / n) * 100

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(confusion_matrix(true_labels, pred_labels, normalize='true')*100,
                ax=ax, annot=True, fmt='.3g', 
                xticklabels=dataloader.dataset.label_map,
                yticklabels=dataloader.dataset.label_map)
    
    fig.savefig(opt.join(save_folder_path, 'cm_best.png'))
                
    return accuracy, true_labels, pred_labels


def training_stgcn_loop(model, mae, train_loader, valid_loader, optimizer, criterion, scheduler, device, args):

    model_args = args.stgcn
    save_folder_path = opt.join(args.save_folder_path, args.exp_name,'weights/')
    os.makedirs(save_folder_path, exist_ok=True)

    start_epoch = 1
    best_val = 0.0

    ## training loop
    for epoch in range(start_epoch, model_args.num_epochs + 1):
        train_loss = train_one_epoch(epoch, model_args.num_epochs, model, mae, optimizer, train_loader, criterion, scheduler, device)
        valid_loss, accuracy, true_labels, pred_labels = valid_one_epoch(model, mae, valid_loader, criterion, device)
        
        is_best = accuracy >= best_val
        best_val = max(accuracy, best_val)
        
        if is_best:
            torch.save(
                {'state_dict': model.state_dict(),
                 'epoch': epoch,
                 'best_val': best_val
                },
                os.path.join(save_folder_path, "best_stgcn_model.pth"),
            )