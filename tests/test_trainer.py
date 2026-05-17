import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from pathlib import Path
import shutil
import re

from wildfire_simulator.models import MK_UNet_Regression
from wildfire_simulator.callbacks import ModelCheckpoint
from wildfire_simulator.trainers import ForwardBurnTrainer

def test_trainer(dataset):
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
    )

    val_loader = DataLoader(
        dataset=dataset,
        batch_size=64,
        shuffle=False,
        drop_last=False,
        num_workers=4,
    )

    model = MK_UNet_Regression(
        in_channels=13,
        out_channels=2,
        channels=[16, 32, 64, 96, 160],
        final_activation='relu'
    )

    checkpoint = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        filename='best-model-{epoch:02d}-{val_loss:.2f}'
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        5e-4,
        weight_decay=1e-4
    )

    # input tensor in training is ForwardBurnProcess with random t
    # min(max(arrival_time), max_t - dt)  
    # can look to test_model for reference on how burner is used
    # dt is the time difference between the input and output tensor
    # output tensor uses burner with t_out = t + dt
    # tensors must be 256 by 256 or 512 by 512 in the last 2 dims
    # can use zero padding
    trainer = ForwardBurnTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.L1Loss(),
        train_loader=train_loader,
        val_loader=val_loader,
        callbacks=[checkpoint],
        epochs=1,
        dt=30,
        max_t=1440
    )

    eval_before = trainer.evaluate()
    assert isinstance(eval_before['val_loss'], float)

    shutil.rmtree('./checkpoints', ignore_errors=True)

    trainer.fit()

    eval_after = trainer.evaluate()
    assert isinstance(eval_after['val_loss'], float)
    assert eval_after['val_loss'] < eval_before['val_loss']

    folder = Path('./checkpoints')
    pattern = re.compile(r"best-model-\d{2}-\d+\.\d{2}\.pt")
    found = any(pattern.fullmatch(p.name) for p in folder.iterdir() if p.is_file())
    assert found
