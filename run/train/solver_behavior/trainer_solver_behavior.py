from pathlib import Path
from collections import deque

import torch
from torch.optim import Adam

from tqdm.auto import tqdm
from ema_pytorch import EMA
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import torch.optim.lr_scheduler as lr_sched

from accelerate import Accelerator, DataLoaderConfiguration

import os
import wandb
import numpy as np

class Trainer(object):
    def __init__(
            self,
            model,
            train_data_loader,
            validation_data_loader,
            *,
            train_batch_size=16,
            gradient_accumulate_every=1,
            train_lr=1e-4,
            train_num_steps=100000,
            ema_update_every=10,
            ema_decay=0.995,
            adam_betas=(0.9, 0.99),
            weight_decay=0.01,
            results_folder='./results',
            project_name='test',
            amp=False,
            mixed_precision_type='fp16',
            split_batches=True,
            max_grad_norm=1.,
            model_params=None,
            curr_datetime,
            accelerator,
            checkpoint_path_to_start=None,
            use_lr_scheduler,
            wandb_run_name,
            previous_run_id=None,
            smoothing_window=5,  # Window size for moving average
            wandb_api_key=None,
    ):
        super().__init__()

        self.accelerator = accelerator

        self.model = model

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.max_grad_norm = max_grad_norm

        self.train_num_steps = train_num_steps

        train_dl = self.accelerator.prepare(train_data_loader)
        self.train_dl = self.cycle(train_dl)
        self.step_per_epoch = len(train_dl)

        val_dl = self.accelerator.prepare(validation_data_loader)
        self.val_dl = val_dl

        # Optimizer and scheduler
        self.opt = Adam(model.parameters(), lr=train_lr, betas=adam_betas)

        self.use_lr_scheduler = use_lr_scheduler
        if self.use_lr_scheduler == "True":
            self.scheduler = ReduceLROnPlateau(self.opt, mode='min', factor=0.5, patience=10, verbose=True, min_lr=1e-6)
        else:
            self.scheduler = None

        if self.accelerator.is_main_process:
            self.ema = EMA(model, beta=ema_decay, update_every=ema_update_every)
            self.ema.to(self.device)

        # Checkpoints
        self.checkpoint_results_folder = Path(f"{results_folder}/checkpoint")
        self.checkpoint_results_folder.mkdir(exist_ok=True, parents=True)

        self.step = 0

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # Best-checkpoint tracking (smoothed validation loss)
        self.best_val_loss = float("inf")
        self.best_milestone_path = None

        # W&B
        self.wandb_results_folder = Path(f"{results_folder}")
        self.wandb_results_folder.mkdir(parents=True, exist_ok=True)
        self.use_wandb = bool(wandb_api_key and str(wandb_api_key).strip())
        if self.accelerator.is_main_process:
            os.environ['WANDB_DIR'] = str(self.wandb_results_folder)
            if self.use_wandb:
                wandb.login(key=wandb_api_key)

            if self.use_wandb:
                run_name = (
                    f"{wandb_run_name}_"
                    f"k_neighbor_{model_params['k_neighbor_to_use']}_"
                    f"training_steps_{model_params['training_steps_limit']}"
                )
                wandb.init(
                    project=project_name,
                    name=run_name,
                    config=model_params,
                    group='multi_gpu_training',
                    job_type='train',
                    resume='allow',
                    id=previous_run_id if previous_run_id is not None else None,
                )

        if checkpoint_path_to_start is not None:
            self.load(checkpoint_path_to_start)

        self.smoothing_window = smoothing_window
        self.val_loss_history = deque(maxlen=smoothing_window)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, model_name):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if self.exists(self.accelerator.scaler) else None,
        }

        torch.save(data, str(self.checkpoint_results_folder / f'{model_name}.pt'))

    def load(self, checkpoint_path_to_start):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(checkpoint_path_to_start), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if self.exists(self.accelerator.scaler) and self.exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    batch = next(self.train_dl)

                    with self.accelerator.autocast():
                        self.curr_epoch_num = self.step // self.step_per_epoch

                        loss = self.model(batch=batch, curr_epoch_num=self.curr_epoch_num)

                        loss = loss.mean() / self.gradient_accumulate_every

                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                if self.accelerator.is_main_process and self.use_wandb:
                    wandb.log({
                        'train total_loss': total_loss,
                        'step': self.step,
                        'learning_rate': self.opt.param_groups[0]['lr']
                    }, commit=True)

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step % self.step_per_epoch == 0 and self.step != 0:
                        milestone = self.step // self.step_per_epoch
                        print(f"Epoch {milestone}")

                        val_loss = self.compute_validation_loss()
                        print(f"val_loss {val_loss}")

                        if self.use_wandb:
                            wandb.log({
                                'val total_loss': val_loss,
                                'epoch': milestone
                            }, commit=True)

                        self.save_best_checkpoint(val_loss, milestone)

                        if self.use_lr_scheduler == "True" and self.scheduler is not None:
                            self.scheduler.step(val_loss)

                pbar.update(1)

        accelerator.print('training complete')

    def compute_moving_average(self):
        return np.mean(self.val_loss_history)

    def save_best_checkpoint(self, val_loss, milestone):
        self.val_loss_history.append(val_loss)

        smoothed_val_loss = self.compute_moving_average()

        if smoothed_val_loss < self.best_val_loss:
            checkpoint_name = f"model-best_validation"
            self.save(checkpoint_name)
            
            milestone_name = f"model-best_validation_epoch-{milestone}"
            self.save(milestone_name)
            new_milestone_path = str(self.checkpoint_results_folder / f'{milestone_name}.pt')
            
            if self.best_milestone_path is not None and os.path.exists(self.best_milestone_path):
                os.remove(self.best_milestone_path)
            
            self.best_val_loss = smoothed_val_loss
            self.best_milestone_path = new_milestone_path
        else:
            pass

    def compute_validation_loss(self):
        self.model.eval()
        self.ema.ema_model.eval()

        total_val_loss = 0.

        with torch.no_grad():
            for batch in self.val_dl:
                val_loss = self.ema.ema_model(batch=batch, curr_epoch_num=self.curr_epoch_num)
                total_val_loss += val_loss.mean().item()

        average_val_loss = total_val_loss / len(self.val_dl)
        return average_val_loss

    def clip_lr(self):

        lr_clip = 1e-6
        for param_group in self.opt.param_groups:
            if param_group['lr'] < lr_clip:
                param_group['lr'] = lr_clip
    
    @staticmethod
    def cycle(dl):
        while True:
            for data in dl:
                yield data
    
    @staticmethod
    def exists(x):
        return x is not None