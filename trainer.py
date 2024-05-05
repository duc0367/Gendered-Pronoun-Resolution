import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler
from transformers import get_linear_schedule_with_warmup
import time
import numpy as np

from dataset import GAPDataset
from config import CFG
from model import GAPModel
from utils import time_since, save_model, load_model
from average_meter import AverageMeter


def train_with_fold(fold: int, train_df: pd.DataFrame, val_df: pd.DataFrame):
    model = GAPModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=CFG.num_warmup_steps,
                                                num_training_steps=len(train_df.index) // CFG.batch_size * CFG.n_epochs)
    train_ds = GAPDataset(train_df)
    val_ds = GAPDataset(val_df)
    train_dl = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=CFG.batch_size)
    model.train()
    scaler = GradScaler(enabled=CFG.enable_scaler)

    losses = AverageMeter()

    for epoch in range(CFG.n_epochs):
        step = 0
        start_time = time.time()
        for batch_idx, (inputs, labels) in enumerate(train_dl):
            inputs = inputs.to(CFG.device, dtype=torch.float32)
            labels = labels.to(CFG.device, dtype=torch.long)
            step += 1
            with torch.autocast(enabled=CFG.enable_scaler, device_type=CFG.device, dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            losses.update(loss.item(), len(inputs))

            if CFG.gradient_accumulation_steps:
                loss = loss / CFG.gradient_accumulation_steps

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), CFG.max_norm)

            if step % CFG.gradient_accumulation_steps == 0 or step == len(train_dl):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if CFG.scheduler:
                    scheduler.step()

            if step % CFG.print_freq == 0 or step == len(train_dl):
                score = validate(model, val_dl, criterion)
                print('Epoch:[{0}/{1}/{2}] '
                      'Loss: {loss:.4f} '
                      'Fold: {fold} '
                      'Lr: {lr:.4f} '
                      'Grad: {grad_norm:.4f} '
                      'Elapsed: {elapsed} '
                      'Score: {score:.4f}'
                      .format(epoch + 1, batch_idx, len(train_dl),
                              loss=loss,
                              fold=fold,
                              lr=scheduler.get_lr()[0],
                              grad_norm=grad_norm,
                              elapsed=time_since(start_time, step / len(train_df)),
                              score=score))

    save_model(model, fold)


def validate(model, val_dl, criterion):
    model.eval()
    losses = []
    for batch_idx, (inputs, labels) in enumerate(val_dl):
        inputs = inputs.to(CFG.device, dtype=torch.float32)
        labels = labels.to(CFG.device, dtype=torch.long)
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        losses.append(loss)
    model.train()
    return sum(losses) / len(losses)


def predict(test_df: pd.DataFrame):
    test_ds = GAPDataset(test_df)
    test_dl = DataLoader(test_ds, batch_size=CFG.batch_size)

    for fold in range(CFG.n_split):
        results = []
        model = load_model(fold)
        model.eval()
        for batch_idx, (inputs, _) in enumerate(test_dl):
            inputs = inputs.to(CFG.device, dtype=torch.float32)
            with torch.no_grad():
                outputs = model(inputs)
            results.append(outputs)
        results = np.concatenate(results)
        test_df[f'prediction_fold_{fold}'] = results
    test_df['prediction'] = test_df[[f'prediction_fold_{fold}' for fold in range(CFG.n_split)]].mean(axis=1)
    return test_df
