import torch
import pytorch_lightning as pl
import wandb

class DDPMImageSampler(pl.Callback):
    def __init__(self, n_samples=1, every_n_epochs=50) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.global_rank == 0:
            if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
                pl_module.eval()
                samples = pl_module.sample_img(n_samples=self.n_samples)
                # -1, 1 -> 0, 1
                samples = (samples + 1) * 0.5

                samples = torch.cat([
                    torch.hstack([img for img in samples[:, idx, ...]])
                    for idx in range(samples.shape[1])
                ], dim=0)
                
                wandb.log({
                    'Generated samples': wandb.Image(
                        samples.detach().cpu().numpy(), 
                        caption='Generated samples (epoch {})'.format(trainer.current_epoch)
                    )
                })
