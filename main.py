import numpy as np
import torch
import pytorch_lightning as pl
import os
from pytorch_lightning.loggers import wandb as wandb_logger
from pytorch_lightning.callbacks import ModelCheckpoint

from modules.preprocessing import BRATSDataModule
from modules.ddpm import DDPM
from modules.diffusion import DiffusionModule
from modules.sampler import ScheduleSampler
from modules.loggers import DDPMImageSampler

os.environ['WANDB_API_KEY'] = 'bdc8857f9d6f7010cff35bcdc0ae9413e05c75e1'

def global_seed(seed, debugging=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    if debugging:
        torch.backends.cudnn.deterministic = True
    
if __name__ == "__main__":
    global_seed(42)
    torch.set_float32_matmul_precision('high')
    
    # logger
    logger = wandb_logger.WandbLogger(
        project='ddpm-brats', 
        name='Train BRATS (My version)'
    )
    
    # data module
    datamodule = BRATSDataModule(
        target_shape=(64, 128, 128),
        n_samples=500,
        modalities=['flair', 'seg'],
        binarize=True,
        train_ratio=1.0,
        npy_path='./data/brats_preprocessed.npy',
        batch_size=16,
        shuffle=True,
        num_workers=6,
        slice_idx=32
    )
    
    sampler = ScheduleSampler(
        T=1000,
        batch_size=16,
        sampler='uniform',
        memory_span=10
    )

    diffusion = DiffusionModule(T=1000, beta_schedule='cosine', input_perturbation=0.1)

    # model
    model = DDPM(
        diffusion=diffusion,
        sampler=sampler,
        in_channels=2,
        out_channels=2,
        image_size=(128, 128),
        timesteps=1000,
        lr = 4.5e-05,
        weight_decay = 1e-07
    )
    
    callbacks = []
    callbacks.append(
        DDPMImageSampler(n_samples=1, every_n_epochs=5)
    )
    
    callbacks.append(
        ModelCheckpoint(
            dirpath='./checkpoints',
            save_top_k=1,
            every_n_epochs=50,
            filename='ddpm-{epoch}'
        )
    )
    
    # training
    trainer = pl.Trainer(
        logger=logger,
        # strategy="ddp",
        # devices=4,
        # num_nodes=2,
        accelerator='gpu',
        precision=32,
        max_epochs=200,
        log_every_n_steps=1,
        enable_progress_bar=True,
        callbacks=callbacks
    )

    trainer.fit(model=model, datamodule=datamodule)
    
    
        
    
