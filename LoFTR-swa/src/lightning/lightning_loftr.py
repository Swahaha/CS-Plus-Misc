from collections import defaultdict
import pprint
from loguru import logger
from pathlib import Path

import torch
import numpy as np
import pytorch_lightning as pl
from matplotlib import pyplot as plt

from src.loftr import LoFTR
from src.loftr.utils.supervision import compute_supervision_coarse, compute_supervision_fine
from src.losses.loftr_loss import LoFTRLoss
from src.optimizers import build_optimizer, build_scheduler
from src.utils.metrics import (
    compute_correspondence_distances,
    aggregate_metrics
)
from src.utils.plotting import make_matching_figures
from src.utils.comm import gather, all_gather
from src.utils.misc import lower_config, flattenList
from src.utils.profiler import PassThroughProfiler


class PL_LoFTR(pl.LightningModule):
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None):
        super().__init__()
        self.save_hyperparameters()  # Save hyperparameters to the checkpoint
        self.config = config 
        _config = lower_config(self.config)
        self.loftr_cfg = lower_config(_config['loftr'])
        self.profiler = profiler or PassThroughProfiler()
        self.n_vals_plot = max(config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1)
        self.epoch_outputs = []  # To collect outputs for on_train_epoch_end
        self.val_outputs = []  # To collect outputs for on_validation_epoch_end

        # Matcher: LoFTR
        self.matcher = LoFTR(config=_config['loftr'])
        self.loss = LoFTRLoss(_config)

        # Pretrained weights
        if pretrained_ckpt:
            state_dict = torch.load(pretrained_ckpt, map_location='cpu')['state_dict']
            self.matcher.load_state_dict(state_dict, strict=True)
            logger.info(f"Load '{pretrained_ckpt}' as pretrained checkpoint")
        
        # Testing
        self.dump_dir = dump_dir
        
    def configure_optimizers(self):
        optimizer = build_optimizer(self, self.config)
        scheduler = build_scheduler(self.config, optimizer)
        return [optimizer], [scheduler]
    
    def optimizer_step(
        self, 
        epoch, 
        batch_idx, 
        optimizer, 
        optimizer_idx, 
        optimizer_closure, 
        on_tpu=False, 
        using_native_amp=False, 
        using_lbfgs=False
    ):
        # Learning rate warm-up
        warmup_step = self.config.TRAINER.WARMUP_STEP
        if self.trainer.global_step < warmup_step:
            if self.config.TRAINER.WARMUP_TYPE == 'linear':
                base_lr = self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
                lr = base_lr + \
                    (self.trainer.global_step / self.config.TRAINER.WARMUP_STEP) * \
                    abs(self.config.TRAINER.TRUE_LR - base_lr)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
            elif self.config.TRAINER.WARMUP_TYPE == 'constant':
                pass
            else:
                raise ValueError(f'Unknown lr warm-up strategy: {self.config.TRAINER.WARMUP_TYPE}')

        # Ensure optimizer_closure is called
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def compute_loss(self, batch):
        self._trainval_inference(batch)
        return batch['loss']
    
    def _trainval_inference(self, batch):
        with self.profiler.profile("Compute coarse supervision"):
            compute_supervision_coarse(batch, self.config)
        
        with self.profiler.profile("LoFTR"):
            self.matcher(batch)
        
        with self.profiler.profile("Compute fine supervision"):
            compute_supervision_fine(batch, self.config)
            
        with self.profiler.profile("Compute losses"):
            self.loss(batch)
    
    def _compute_metrics(self, batch):
        compute_correspondence_distances(batch)

        distances = batch['correspondence_distances']
        assert isinstance(distances, list), "distances should be a list"
        assert all(isinstance(d, list) for d in distances), "Each element of distances should be a list"

        if not distances:
            logger.error("No distances found after computation.")

        # Debug: Log some sample distances
        if distances and distances[0]:
            logger.debug(f"Sample distances: {distances[0][:5]}")

        mean_distances = [torch.tensor(d).float().mean().item() for d in distances if d]
        max_distances = [torch.tensor(d).float().max().item() for d in distances if d]
        min_distances = [torch.tensor(d).float().min().item() for d in distances if d]

        ret_dict = {
            'mean_distance': sum(mean_distances) / len(mean_distances) if mean_distances else 0,
            'max_distance': max(max_distances) if max_distances else 0,
            'min_distance': min(min_distances) if min_distances else 0
        }

        return ret_dict, distances

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        avg_loss = torch.stack([x['loss'] for x in self.epoch_outputs]).mean()
        if self.trainer.global_rank == 0:
            self.logger.experiment.add_scalar(
                'train/avg_loss_on_epoch', avg_loss,
                global_step=self.current_epoch)
        self.epoch_outputs = []  # Clear the list for the next epoch
    
    def validation_step(self, batch, batch_idx):
        ret_dict, distances = self._compute_metrics(batch)
        self.log_dict(ret_dict, on_step=False, on_epoch=True, sync_dist=True)

        assert isinstance(distances, list), "distances should be a list"
        assert all(isinstance(d, list) for d in distances), "Each element of distances should be a list"

        # Debug: Log the structure of distances
        if distances and distances[0]:
            logger.debug(f"Distances in validation_step: {distances[0][:5]}")

        loss = self.compute_loss(batch)
        self.val_outputs.append({'loss_scalars': {'loss': loss}, 'metrics': ret_dict, 'correspondence_distances': distances})

        return loss

    def on_validation_epoch_end(self):
        multi_outputs = [self.val_outputs]
        multi_val_metrics = defaultdict(list)
        
        for valset_idx, outputs in enumerate(multi_outputs):
            if getattr(self.trainer, 'running_sanity_check', False):
                return

            cur_epoch = self.trainer.current_epoch

            _loss_scalars = [o['loss_scalars'] for o in outputs]
            loss_scalars = {k: flattenList(all_gather([_ls[k] for _ls in _loss_scalars])) for k in _loss_scalars[0]}

            _metrics = [o['metrics'] for o in outputs]
            logger.info(f"_metrics: {_metrics}")

            if _metrics and _metrics[0]:
                metrics = {}
                for k in _metrics[0]:
                    values = [v if isinstance(v, list) else [v] for v in [_me[k] for _me in _metrics]]
                    values = flattenList(values)
                    gathered = all_gather(values)
                    if not isinstance(gathered, list):
                        gathered = [gathered]
                    metrics[k] = flattenList(gathered)
                
                if 'identifiers' not in metrics:
                    metrics['identifiers'] = [i for i in range(len(metrics['mean_distance']))]
                
                if 'correspondence_distances' not in metrics:
                    metrics['correspondence_distances'] = flattenList([o['correspondence_distances'] for o in outputs])

                # Ensure all elements are lists and log types for debugging
                correspondence_distances = []
                for sublist in metrics['correspondence_distances']:
                    if not isinstance(sublist, list):
                        logger.warning(f"Type before conversion: {type(sublist)}")
                        sublist = [sublist]
                    logger.debug(f"Type after conversion: {type(sublist)}")
                    correspondence_distances.append(sublist)

                # Flatten and validate distances
                flat_distances = [dist for sublist in correspondence_distances for dist in sublist]
                flat_distances = [float(d) for d in flat_distances if isinstance(d, (int, float))]

                # logger.debug(f"Flat distances: {flat_distances}")

                metrics['correspondence_distances'] = flat_distances

                val_metrics_4tb = aggregate_metrics(metrics, self.config.TRAINER.EPI_ERR_THR)
                for thr in [5, 10, 20]:
                    multi_val_metrics[f'auc@{thr}'].append(val_metrics_4tb.get(f'auc@{thr}', 0))
            
                if self.trainer.global_rank == 0:
                    for k, v in loss_scalars.items():
                        mean_v = torch.stack(v).mean()
                        self.logger.experiment.add_scalar(f'val_{valset_idx}/avg_{k}', mean_v, global_step=cur_epoch)

                    for k, v in val_metrics_4tb.items():
                        self.logger.experiment.add_scalar(f"metrics_{valset_idx}/{k}", v, global_step=cur_epoch)
                
                    plt.close('all')

        for thr in [5, 10, 20]:
            if multi_val_metrics[f'auc@{thr}']:
                self.log(f'auc@{thr}', torch.tensor(np.mean(multi_val_metrics[f'auc@{thr}'])), sync_dist=True)

        self.val_outputs = []

    def test_step(self, batch, batch_idx):
        with self.profiler.profile("LoFTR"):
            self.matcher(batch)

        ret_dict, rel_pair_names = self._compute_metrics(batch)

        with self.profiler.profile("dump_results"):
            if self.dump_dir is not None:
                keys_to_save = {'mkpts0_f', 'mkpts1_f', 'mconf', 'epi_errs'}
                pair_names = list(zip(*batch['pair_names']))
                bs = batch['image0'].shape[0]
                dumps = []
                for b_id in range(bs):
                    item = {}
                    mask = batch['m_bids'] == b_id
                    item['pair_names'] = pair_names[b_id]
                    item['identifier'] = '#'.join(rel_pair_names[b_id])
                    for key in keys_to_save:
                        item[key] = batch[key][mask].cpu().numpy()
                    for key in ['R_errs', 't_errs', 'inliers']:
                        item[key] = batch[key][b_id]
                    dumps.append(item)
                ret_dict['dumps'] = dumps

        return ret_dict

    def test_epoch_end(self, outputs):
        _metrics = [o['metrics'] for o in outputs]
        metrics = {k: flattenList(gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}

        if self.dump_dir is not None:
            Path(self.dump_dir).mkdir(parents=True, exist_ok=True)
            _dumps = flattenList([o['dumps'] for o in outputs])
            dumps = flattenList(gather(_dumps))
            logger.info(f'Prediction and evaluation results will be saved to: {self.dump_dir}')

        if self.trainer.global_rank == 0:
            print(self.profiler.summary())
            val_metrics_4tb = aggregate_metrics(metrics, self.config.TRAINER.EPI_ERR_THR)
            logger.info('\n' + pprint.pformat(val_metrics_4tb))
            if self.dump_dir is not None:
                np.save(Path(self.dump_dir) / 'LoFTR_pred_eval', dumps)
