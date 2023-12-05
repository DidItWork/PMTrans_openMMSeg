# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .utils import get_class_weight, weight_reduce_loss


@MODELS.register_module()
class PMLabelLoss(nn.Module):
    """Mix Source Target Loss from PMTrans.

    Args:
        
    """

    def __init__(self,
                 loss_name='loss_pm_label',):
        super().__init__()

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
    

@MODELS.register_module()
class PMFeatureLoss(nn.Module):
    """Mix Source Target Loss from PMTrans.

    Args:
        
    """

    def __init__(self,
                 loss_name='loss_pm_feature',):
        super().__init__()

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
