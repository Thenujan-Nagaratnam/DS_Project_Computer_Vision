import numpy as np
import torch

from tqdm import tqdm

from utils import AverageMeter
import utils as utilities

def ArcFace_criterion(logits_m, target, margins):
    arc = utilities.ArcFaceLossAdaptiveMargin(margins=margins, s=CFG.s, crit=CFG.crit)
    loss_m = arc(logits_m, target, CFG.n_classes)
    return loss_m

    
def train(model, train_loader, optimizer, scaler, scheduler, epoch):
    model.train()
    loss_metrics = utilities.AverageMeter()
    criterion = ArcFace_criterion

    tmp = np.sqrt(1 / np.sqrt(value_counts))
    margins = (tmp - tmp.min()) / (tmp.max() - tmp.min()) * CFG.m + CFG.m_min
        
    bar = tqdm(train_loader)
    for step, data in enumerate(bar):
        step += 1
        images = data['images'].to(CFG.device, dtype=torch.float)
        labels = data['labels'].to(CFG.device)
        batch_size = labels.size(0)

        with torch.cuda.amp.autocast(enabled=CFG.autocast):
            outputs, features = model(images)

        loss = criterion(outputs, labels, margins)
        loss_metrics.update(loss.item(), batch_size)
        loss = loss / CFG.acc_steps
        scaler.scale(loss).backward()

        if step % CFG.acc_steps == 0 or step == len(bar):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            CFG.global_step += 1
                        
        lrs = utilities.get_lr_groups(optimizer.param_groups)

        loss_avg = loss_metrics.avg

        bar.set_postfix(loss=loss_avg, epoch=epoch, lrs=lrs, step=CFG.global_step)
    
@torch.no_grad()
def val(model, valid_loader):
    model.eval() 

    all_embeddings = []
    all_labels = [] 

    for data in tqdm(valid_loader):
        images = data['images'].to(CFG.device, dtype=torch.float)
        labels = data['labels'].to(CFG.device)

        _, embeddings = model(images)

        all_embeddings.append(embeddings.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())


    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_embeddings, all_labels
