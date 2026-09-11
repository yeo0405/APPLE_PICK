"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import re
import copy

from ._config import BaseConfig
from .workspace import create
from .yaml_utils import load_config, merge_config, merge_dict


class YAMLConfig(BaseConfig):
    def __init__(self, cfg_path: str, **kwargs) -> None:
        super().__init__()

        # ============================================================
        # Config path
        # ============================================================
        #
        # Make cfg_path absolute so that all relative paths in the
        # YAML file can be resolved relative to the YAML file itself,
        # instead of depending on os.getcwd().
        #
        self.cfg_path = os.path.abspath(os.path.expanduser(cfg_path))
        self.cfg_dir = os.path.dirname(self.cfg_path)

        print(f"[YAMLConfig] Config file : {self.cfg_path}")
        print(f"[YAMLConfig] Config dir  : {self.cfg_dir}")

        # ============================================================
        # Load YAML
        # ============================================================

        cfg = load_config(self.cfg_path)
        cfg = merge_dict(cfg, kwargs)

        self.yaml_cfg = copy.deepcopy(cfg)

        # ============================================================
        # Resolve paths
        # ============================================================
        #
        # All relative paths in the YAML are interpreted relative to
        # the directory containing the YAML file.
        #
        self._resolve_config_paths()

        # ============================================================
        # Copy BaseConfig attributes
        # ============================================================

        for k in super().__dict__:
            if not k.startswith('_') and k in self.yaml_cfg:
                self.__dict__[k] = self.yaml_cfg[k]

    # ================================================================
    # Path utilities
    # ================================================================

    def _resolve_path(self, path):
        """
        Resolve a path relative to the YAML config directory.

        Absolute paths:
            /home/user/file
        are kept unchanged.

        Relative paths:
            ./xxx
            ../xxx
        are resolved relative to self.cfg_dir.
        """

        if path is None:
            return None

        path = os.path.expanduser(str(path))

        if os.path.isabs(path):
            return os.path.normpath(path)

        return os.path.normpath(
            os.path.join(self.cfg_dir, path)
        )

    def _resolve_config_paths(self):
        """
        Resolve all RACE-6D paths that are used by the config.

        Paths in race6d_r50vd_pose_rgbd.yml are interpreted relative
        to the YAML file location.
        """

        # ------------------------------------------------------------
        # output_dir
        # ------------------------------------------------------------

        if 'output_dir' in self.yaml_cfg:
            self.yaml_cfg['output_dir'] = self._resolve_path(
                self.yaml_cfg['output_dir']
            )

        # ------------------------------------------------------------
        # coco_path
        # ------------------------------------------------------------

        if 'coco_path' in self.yaml_cfg:
            self.yaml_cfg['coco_path'] = self._resolve_path(
                self.yaml_cfg['coco_path']
            )

        # ------------------------------------------------------------
        # category_file
        # ------------------------------------------------------------

        if 'category_file' in self.yaml_cfg:
            self.yaml_cfg['category_file'] = self._resolve_path(
                self.yaml_cfg['category_file']
            )

        # ------------------------------------------------------------
        # Optional keypoints cache
        #
        # This does not change race6d_decoder_dqe.py directly.
        # It is added so the resolved cache path is available in the
        # global YAML configuration if needed elsewhere.
        # ------------------------------------------------------------

        if 'keypoints_path' in self.yaml_cfg:
            self.yaml_cfg['keypoints_path'] = self._resolve_path(
                self.yaml_cfg['keypoints_path']
            )

        # ------------------------------------------------------------
        # Print resolved paths for debugging
        # ------------------------------------------------------------

        print("[YAMLConfig] Resolved paths:")

        if 'output_dir' in self.yaml_cfg:
            print(
                f"  output_dir    = "
                f"{self.yaml_cfg['output_dir']}"
            )

        if 'coco_path' in self.yaml_cfg:
            print(
                f"  coco_path     = "
                f"{self.yaml_cfg['coco_path']}"
            )

        if 'category_file' in self.yaml_cfg:
            print(
                f"  category_file = "
                f"{self.yaml_cfg['category_file']}"
            )

        if 'keypoints_path' in self.yaml_cfg:
            print(
                f"  keypoints     = "
                f"{self.yaml_cfg['keypoints_path']}"
            )

    # ================================================================
    # Global config
    # ================================================================

    @property
    def global_cfg(self):
        return merge_config(
            self.yaml_cfg,
            inplace=False,
            overwrite=False
        )

    # ================================================================
    # Model
    # ================================================================

    @property
    def model(self) -> torch.nn.Module:
        if self._model is None and 'model' in self.yaml_cfg:
            self._model = create(
                self.yaml_cfg['model'],
                self.global_cfg
            )
        return super().model

    # ================================================================
    # Postprocessor
    # ================================================================

    @property
    def postprocessor(self) -> torch.nn.Module:
        if self._postprocessor is None and 'postprocessor' in self.yaml_cfg:
            self._postprocessor = create(
                self.yaml_cfg['postprocessor'],
                self.global_cfg
            )
        return super().postprocessor

    # ================================================================
    # Criterion
    # ================================================================

    @property
    def criterion(self) -> torch.nn.Module:
        if self._criterion is None and 'criterion' in self.yaml_cfg:
            self._criterion = create(
                self.yaml_cfg['criterion'],
                self.global_cfg
            )
        return super().criterion

    # ================================================================
    # Optimizer
    # ================================================================

    @property
    def optimizer(self) -> optim.Optimizer:
        if self._optimizer is None and 'optimizer' in self.yaml_cfg:
            params = self.get_optim_params(
                self.yaml_cfg['optimizer'],
                self.model
            )

            self._optimizer = create(
                'optimizer',
                self.global_cfg,
                params=params
            )

        return super().optimizer

    # ================================================================
    # LR Scheduler
    # ================================================================

    @property
    def lr_scheduler(self) -> optim.lr_scheduler.LRScheduler:
        if self._lr_scheduler is None and 'lr_scheduler' in self.yaml_cfg:
            self._lr_scheduler = create(
                'lr_scheduler',
                self.global_cfg,
                optimizer=self.optimizer
            )

            print(
                f'Initial lr: '
                f'{self._lr_scheduler.get_last_lr()}'
            )

        return super().lr_scheduler

    # ================================================================
    # LR Warmup Scheduler
    # ================================================================

    @property
    def lr_warmup_scheduler(self) -> optim.lr_scheduler.LRScheduler:
        if (
            self._lr_warmup_scheduler is None
            and 'lr_warmup_scheduler' in self.yaml_cfg
        ):
            self._lr_warmup_scheduler = create(
                'lr_warmup_scheduler',
                self.global_cfg,
                lr_scheduler=self.lr_scheduler
            )

        return super().lr_warmup_scheduler

    # ================================================================
    # Train DataLoader
    # ================================================================

    @property
    def train_dataloader(self) -> DataLoader:
        if (
            self._train_dataloader is None
            and 'train_dataloader' in self.yaml_cfg
        ):
            self._train_dataloader = self.build_dataloader(
                'train_dataloader'
            )

        return super().train_dataloader

    # ================================================================
    # Validation DataLoader
    # ================================================================

    @property
    def val_dataloader(self) -> DataLoader:
        if (
            self._val_dataloader is None
            and 'val_dataloader' in self.yaml_cfg
        ):
            self._val_dataloader = self.build_dataloader(
                'val_dataloader'
            )

        return super().val_dataloader

    # ================================================================
    # EMA
    # ================================================================

    @property
    def ema(self) -> torch.nn.Module:
        if (
            self._ema is None
            and self.yaml_cfg.get('use_ema', False)
        ):
            self._ema = create(
                'ema',
                self.global_cfg,
                model=self.model
            )

        return super().ema

    # ================================================================
    # AMP scaler
    # ================================================================

    @property
    def scaler(self):
        if (
            self._scaler is None
            and self.yaml_cfg.get('use_amp', False)
        ):
            self._scaler = create(
                'scaler',
                self.global_cfg
            )

        return super().scaler

    # ================================================================
    # Evaluator
    # ================================================================

    @property
    def evaluator(self):
        if self._evaluator is None and 'evaluator' in self.yaml_cfg:

            if self.yaml_cfg['evaluator']['type'] == 'CocoEvaluator':

                from ..data import get_coco_api_from_dataset

                base_ds = get_coco_api_from_dataset(
                    self.val_dataloader.dataset
                )

                self._evaluator = create(
                    'evaluator',
                    self.global_cfg,
                    coco_gt=base_ds
                )

            else:
                raise NotImplementedError(
                    f"{self.yaml_cfg['evaluator']['type']}"
                )

        return super().evaluator

    # ================================================================
    # Optimizer parameter groups
    # ================================================================

    @staticmethod
    def get_optim_params(cfg: dict, model: nn.Module):
        """
        E.g.:

            ^(?=.*a)(?=.*b).*$

        means including a and b

            ^(?=.*(?:a|b)).*$

        means including a or b

            ^(?=.*a)(?!.*b).*$

        means including a, but not b
        """

        assert 'type' in cfg, ''

        cfg = copy.deepcopy(cfg)

        if 'params' not in cfg:
            return model.parameters()

        assert isinstance(cfg['params'], list), ''

        param_groups = []
        visited = []

        for pg in cfg['params']:

            pattern = pg['params']

            params = {
                k: v
                for k, v in model.named_parameters()
                if v.requires_grad
                and len(re.findall(pattern, k)) > 0
            }

            pg['params'] = params.values()

            param_groups.append(pg)

            visited.extend(list(params.keys()))

        names = [
            k
            for k, v in model.named_parameters()
            if v.requires_grad
        ]

        if len(visited) < len(names):

            unseen = set(names) - set(visited)

            params = {
                k: v
                for k, v in model.named_parameters()
                if v.requires_grad
                and k in unseen
            }

            param_groups.append({
                'params': params.values()
            })

            visited.extend(list(params.keys()))

        assert len(visited) == len(names), ''

        return param_groups

    # ================================================================
    # Batch size
    # ================================================================

    @staticmethod
    def get_rank_batch_size(cfg):

        assert (
            ('total_batch_size' in cfg or 'batch_size' in cfg)
            and not (
                'total_batch_size' in cfg
                and 'batch_size' in cfg
            )
        ), '`batch_size` or `total_batch_size` should be choosed one'

        total_batch_size = cfg.get(
            'total_batch_size',
            None
        )

        if total_batch_size is None:

            bs = cfg.get('batch_size')

        else:

            from ..misc import dist_utils

            assert (
                total_batch_size
                % dist_utils.get_world_size()
                == 0
            ), (
                'total_batch_size should be divisible '
                'by world size'
            )

            bs = (
                total_batch_size
                // dist_utils.get_world_size()
            )

        return bs

    # ================================================================
    # DataLoader
    # ================================================================

    def build_dataloader(self, name: str):

        bs = self.get_rank_batch_size(
            self.yaml_cfg[name]
        )

        global_cfg = self.global_cfg

        if 'total_batch_size' in global_cfg[name]:

            # pop unexpected key for dataloader init
            _ = global_cfg[name].pop(
                'total_batch_size'
            )

        print(
            f'building {name} '
            f'with batch_size={bs}...'
        )

        loader = create(
            name,
            global_cfg,
            batch_size=bs
        )

        loader.shuffle = self.yaml_cfg[name].get(
            'shuffle',
            False
        )

        return loader