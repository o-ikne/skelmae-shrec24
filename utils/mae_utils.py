import torch

from tqdm import tqdm
import os
import os.path as opt


def train_one_epoch(epoch, num_epochs, model, dataloader, optimizer, scheduler, device):
    model.train()

    pbar = tqdm(dataloader, total=len(dataloader))
    train_loss = 0

    for d in pbar:
        sequence = d['Sequence'].to(device)
        adj_mat = d['AM'].to(device)

        optimizer.zero_grad()               
        *_, loss = model(sequence, adj_mat)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pbar.set_description(f'[%.3g/%.3g] train loss. %.2f' % (epoch, num_epochs, train_loss))

        break
        
    if scheduler is not None:
        scheduler.step()
    
    return train_loss

def valid_one_epoch(model, dataloader, device):
    valid_loss = 0.0
    model.eval()
    with torch.no_grad():
        pbar = tqdm(dataloader, total=len(dataloader))
        for d in pbar:
            sequence = d['Sequence'].to(device)
            adj_mat = d['AM'].to(device) 

            *_, loss = model(sequence, adj_mat)
            valid_loss += loss.item()
            
            desc = '[VALID] valid loss. %.2f' % (valid_loss)
            pbar.set_description(desc)

            break
                
    return valid_loss


def training_mae_loop(model, train_loader, valid_loader, optimizer, scheduler, device, args):

    save_folder_path = opt.join(args.save_folder_path, args.exp_name,'weights/')
    os.makedirs(save_folder_path, exist_ok=True)
    
    ## TRAINING
    start_epoch = 1
    best_val = float("inf")

    ## training loop
    for epoch in range(start_epoch, args.mae.num_epochs + 1):
        train_loss = train_one_epoch(epoch, args.mae.num_epochs, model, train_loader, optimizer, scheduler, device)
        valid_loss = valid_one_epoch(model, valid_loader, device)
        
        is_best = valid_loss < best_val
        best_val = min(valid_loss, best_val)
        
        if is_best:
            torch.save(
                {'state_dict': model.state_dict(),
                 'epoch': epoch,
                 'best_val': best_val
                },
                opt.join(save_folder_path, "best_mae_model.pth"),
            )