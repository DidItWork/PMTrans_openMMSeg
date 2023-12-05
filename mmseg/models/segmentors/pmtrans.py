# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import print_log
from torch import Tensor
import torch.distributions as dists

from mmseg.registry import MODELS
from mmseg.utils import (ForwardResults, ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .base import BaseSegmentor

import os

# os.environ['CUDA_LAUNCH_BLOCKING']='1'


@MODELS.register_module()
class PMTrans(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.

    1. The ``loss`` method is used to calculate the loss of model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2) Call the decode head loss function to forward decode head model and
    calculate losses.

    .. code:: text

     loss(): extract_feat() -> _decode_head_forward_train() -> _auxiliary_head_forward_train (optional)
     _decode_head_forward_train(): decode_head.loss()
     _auxiliary_head_forward_train(): auxiliary_head.loss (optional)

    2. The ``predict`` method is used to predict segmentation results,
    which includes two steps: (1) Run inference function to obtain the list of
    seg_logits (2) Call post-processing function to obtain list of
    ``SegDataSample`` including ``pred_sem_seg`` and ``seg_logits``.

    .. code:: text

     predict(): inference() -> postprocess_result()
     infercen(): whole_inference()/slide_inference()
     whole_inference()/slide_inference(): encoder_decoder()
     encoder_decoder(): extract_feat() -> decode_head.predict()

    3. The ``_forward`` method is used to output the tensor by running the model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2)Call the decode head forward function to forward decode head model.

    .. code:: text

     _forward(): extract_feat() -> _decode_head.forward()

    Args:

        backbone (ConfigType): The config for the backnone of segmentor.
        decode_head (ConfigType): The config for the decode head of segmentor.
        neck (OptConfigType): The config for the neck of segmentor.
            Defaults to None.
        auxiliary_head (OptConfigType): The config for the auxiliary head of
            segmentor. Defaults to None.
        train_cfg (OptConfigType): The config for training. Defaults to None.
        test_cfg (OptConfigType): The config for testing. Defaults to None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
        pretrained (str, optional): The path for pretrained model.
            Defaults to None.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.
    """  # noqa: E501

    def __init__(self,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

        self.s_dist_alpha = nn.Parameter(Tensor([1]))
        self.s_dist_beta = nn.Parameter(Tensor([1]))
        self.super_ratio = nn.Parameter(Tensor([-2]))
        self.unsuper_ratio = nn.Parameter(Tensor([-2]))

    def _init_decode_head(self, decode_head: ConfigType) -> None:
        """Initialize ``decode_head``"""
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def _init_auxiliary_head(self, auxiliary_head: ConfigType) -> None:
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        # print(self.backbone(inputs)[0].shape)
        x = self.backbone(inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""

        x = self.extract_feat(inputs)
        seg_logits = self.decode_head.predict(x, batch_img_metas,
                                              self.test_cfg)

        return seg_logits

    def _decode_head_forward_train(self, inputs: List[Tensor],
                                   data_samples: SampleList, features=False) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.loss(inputs, data_samples,
                                            self.train_cfg)
        
        # for key,val in loss_decode.items():
        #     print("decode loss",key,val)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _auxiliary_head_forward_train(self, inputs: List[Tensor],
                                      data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(inputs, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(inputs, data_samples,
                                                self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses
    


    #-----------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------
    #Mix up stuff
    #Mix up stuff
    #Mix up stuff
    #-----------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------



    def softplus(self, x):
        return torch.log(1+torch.exp(x))
    
    def mix_token(self, s_tokens, t_tokens, s_lambda):

        s_tokens = torch.einsum('BCHW,BHW -> BCHW', s_tokens, s_lambda)
        t_tokens = torch.einsum('BCHW,BHW -> BCHW', t_tokens, 1-s_lambda)

        m_tokens = s_tokens+t_tokens

        return m_tokens
    
    def mix_lambda_atten(self, s_scores, t_scores, s_lambda):
        
        #Returns a list of lambdas according to the number of layers of attention scores

        s_lambdas = []

        s_lambda = nn.Flatten()(s_lambda)
        
        for i in range(len(s_scores)):

            num_patch = s_scores[i].shape[-1] #'Source' patches in attention

            s_l = nn.AdaptiveAvgPool1d(num_patch)(s_lambda)
            
            t_l = 1-s_l

            s_lambda = torch.einsum('BLK,BK->BL',s_scores[i], s_l)
            t_lambda = torch.einsum('BLK,BK->BL',t_scores[i], t_l)
            s_lambdas.append(s_lambda/(s_lambda+t_lambda))

            # print(s_lambdas[i])

        return s_lambdas

    def cosine_distance(self,s_logits, t_logits):
        temp_matrix = []
        for l in range(len(s_logits)):

            s_logit = nn.Flatten()(s_logits[l].permute(1,0,2,3))
            t_logit = nn.Flatten()(t_logits[l].permute(1,0,2,3))
            temp_matrix.append(torch.mm(s_logit.t(),t_logit))

            norm_s = torch.norm(s_logit,p=2,dim=0).unsqueeze(0)
            norm_t = torch.norm(t_logit,p=2,dim=0).unsqueeze(0)
            norm_matrix = torch.mm(norm_s.t(),norm_t)
            norm_matrix = torch.maximum(norm_matrix,torch.ones_like(norm_matrix).cuda()*1e-6)
            temp_matrix[l]/=norm_matrix
                
        return temp_matrix


    def mixup_supervised_dis(self, preds, s_label, lam):

        mixup_losses = torch.tensor(0.,requires_grad=True).cuda()

        s_label_tensor = []

        for label in s_label:

            label_onehot = F.one_hot(label.gt_sem_seg.data.squeeze(),num_classes=256)[:,:,:self.num_classes]

            s_label_tensor.append(label_onehot.float())
        
        s_label_tensor = torch.stack(s_label_tensor, dim=0).permute(3,0,1,2)

        s_label_tensor = nn.AdaptiveAvgPool1d(preds[0].shape[-1])(nn.Flatten()(s_label_tensor))

        label_norm = torch.norm(s_label_tensor,p=2,dim=0).unsqueeze(0)

        norm_matrix = torch.mm(label_norm.t(),label_norm)

        norm_matrix = torch.maximum(norm_matrix, torch.ones_like(norm_matrix)*1e-6)

        slabel = torch.mm(s_label_tensor.t(),s_label_tensor).unsqueeze(0)

        slabel/=norm_matrix

        for p in range(len(preds)):

            label = nn.AdaptiveAvgPool2d(preds[p].shape[-2:])(slabel)

            lambda_ = nn.Flatten(start_dim=0)(lam[p]).unsqueeze(0)

            mixup_loss = -torch.sum(label.squeeze(0)*F.log_softmax(preds[p],dim=1),dim=1).unsqueeze(0)

            mixup_loss /= torch.maximum(torch.sum(label,dim=1),torch.ones_like(mixup_loss)*1e-6)

            mixup_losses += torch.mm(lambda_,mixup_loss.t()).item()/lambda_.shape[-1]
        
        # print('Supervised',mixup_losses/len(preds))
        
        return mixup_losses/len(preds)


    def mixup_unsupervised_dis(self, preds, lam):

        mixup_losses = torch.tensor(0.,requires_grad=True).cuda()
 
        for p in range(len(preds)):

            label = torch.eye(preds[p].shape[0]).cuda()

            mixup_loss = -torch.sum(label*F.log_softmax(preds[p],dim=1),dim=1).unsqueeze(0)

            lambda_ = nn.Flatten(start_dim=0)(lam[p]).unsqueeze(0)

            mixup_losses += torch.mm(lambda_,mixup_loss.t()).item()/lambda_.shape[-1]
        
        # print('Unsupervised',mixup_losses/len(preds))

        return mixup_losses/len(preds) #N H x W
    
    def mixup_soft_ce(self, pred, labels, weight, lam):

        pred_labels = []

        for label in labels:
            pred_labels.append(label.gt_sem_seg.data.squeeze())
        
        pred_labels = torch.stack(pred_labels)

        loss = torch.nn.CrossEntropyLoss(reduction='none',weight=weight,ignore_index=255)(pred, pred_labels)
        
        # count = torch.unique(pred_labels,return_counts=True)

        # if count[0][-1]==255:

        #     count[0][-1] = 0

        #     count[1][-1] = 0

        # count = torch.mm(weight[count[0]].unsqueeze(0),count[1].unsqueeze(-1).float()).item()

        weight = torch.cat((weight,torch.zeros(256-self.num_classes).cuda()),0)
        
        count = torch.sum(torch.mul(lam,weight[pred_labels]))

        loss = torch.mean(lam)*(torch.sum(torch.mul(loss, lam))/count)

        return loss
    
    def mix_source_target(self, s_token, t_token, s_lambda, t_lambda,
                          target_label, infer_label, s_logits, t_logits,
                          s_scores, t_scores, weight_tgt, weight_src,
                          source_masks, target_masks):
        
        # print("Mixing tokens")
        m_s_t_token = self.mix_token(s_token, t_token, s_lambda)

        # print("Forward Pass for intermediate domain")
        m_s_t_logits, m_s_t_p, _ = self.backbone.forward_features(m_s_t_token, p_in=True)
        
        # print("Generating intermediate domain predictions")
        m_s_t_pred = self.decode_head.forward(m_s_t_logits)

        m_s_t_pred = F.interpolate(m_s_t_pred,infer_label[0].gt_sem_seg.data.shape[-2:],mode='bilinear',align_corners=True)

        m_s_t_pred = nn.Softmax(dim=1)(m_s_t_pred)

        # print("Applying paddings masks on m_s_t_logits")

        # m_s_t_logits = m_s_t_logits[-1:]

        for i in range(len(m_s_t_logits)):

            scale = int(target_masks.shape[-1]/m_s_t_logits[i].shape[-1])

            h_indexes = scale*torch.arange(target_masks.shape[-1]//scale,dtype=torch.int).cuda()
            v_indexes = scale*torch.arange(target_masks.shape[-2]//scale,dtype=torch.int).cuda()

            mask = torch.logical_or(source_masks,target_masks).cuda()

            mask = torch.index_select(torch.index_select(mask,2,h_indexes),1,v_indexes)

            # print(mask, mask.shape)

            m_s_t_logits[i] = torch.mul(m_s_t_logits[i],mask.unsqueeze(1))

        # print("Mixing Lambda Attention")
        s_lambda = self.mix_lambda_atten(s_scores, t_scores, s_lambda) #B x HW

        # print("lambda", s_lambda)

        t_lambda = []

        for i in range(len(s_lambda)):
            #Reshape s_lambda to layers x B x H x W
            lam_shape = (s_lambda[i].shape[0],*m_s_t_logits[i].shape[-2:])
            s_lambda[i] = torch.reshape(s_lambda[i],lam_shape)

            #Calculate t_lambda
            t_lambda.append(1-s_lambda[i])
        

        # print("Cosine Distance")
        m_s_t_s = self.cosine_distance(m_s_t_logits[-1:],s_logits[-1:])

        m_s_t_s_similarity = self.mixup_supervised_dis(m_s_t_s, infer_label[-1:] ,s_lambda[-1:])

        m_s_t_t = self.cosine_distance(m_s_t_logits[-1:],t_logits[-1:])

        m_s_t_t_similarity = self.mixup_unsupervised_dis(m_s_t_t, t_lambda[-1:])

        #resizing and concatenating feature_space_loss across layers to be same size as segmentation map

        super_feature_space_loss = self.softplus(self.super_ratio)*(m_s_t_s_similarity)
        unsuper_feature_space_loss = self.softplus(self.super_ratio)*m_s_t_t_similarity

        #resizing and concatenating lambda to be same size as segmentation map

        length = len(s_lambda)
        
        s_lam = F.interpolate(s_lambda[0].unsqueeze(0),size=infer_label[0].gt_sem_seg.data.shape[-2:],mode='bilinear',align_corners=True).squeeze(0)

        for l in range(1,len(s_lambda)):
            s_lam += F.interpolate(s_lambda[l].unsqueeze(0),size=infer_label[0].gt_sem_seg.data.shape[-2:],mode='bilinear',align_corners=True).squeeze(0)
        
        s_lam /= length

        t_lam = 1 - s_lam

        # print("Mixup soft CE")
        super_m_s_t_s_loss = self.mixup_soft_ce(m_s_t_pred, infer_label, weight_src, s_lam)
        unsuper_m_s_t_loss = self.mixup_soft_ce(m_s_t_pred, target_label, weight_tgt, t_lam)

        super_label_space_loss = self.softplus(self.unsuper_ratio)*super_m_s_t_s_loss
        unsuper_label_space_loss = self.softplus(self.unsuper_ratio)*unsuper_m_s_t_loss

        #Losses

        f_loss_dict = dict()
        l_loss_dict = dict()

        # f_losses = dict(loss_pm_feature=super_feature_space_loss, loss_pm_feature=unsuper_feature_space_loss)
        # l_losses = dict(loss_pm_label=label_space_loss)
        # l_losses = dict()
        # f_losses = dict()

        f_loss_dict.update(add_prefix(dict(loss_pm_feature=super_feature_space_loss), 'super'))
        f_loss_dict.update(add_prefix(dict(loss_pm_feature=unsuper_feature_space_loss), 'unsuper'))
        l_loss_dict.update(add_prefix(dict(loss_ce=super_label_space_loss), 'super'))
        l_loss_dict.update(add_prefix(dict(loss_ce=unsuper_label_space_loss), 'unsuper'))

        return f_loss_dict, l_loss_dict

    def loss(self, inputs: Tensor, data_samples: SampleList, targets: Tensor, target_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of target and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.
            targets: Unlabelled target images.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        losses = dict()

        # x = self.extract_feat(target)

        """
        Additional losses from patch-mixing

        Backbone produces feature maps of sizes: 1/4, 1/8, 1/16, 1/32
        Attention sr_ratios=[8, 4, 2, 1]

        logits: len(self.out_indices) x B x C x H x W
        patch: B x C x H X W
        attns: len(self.out_indices) x B x HW x HW
        """

        #Create padding masks

        # print(self.s_dist_alpha, self.s_dist_beta, self.super_ratio, self.unsuper_ratio)

        source_masks = []
        target_masks = []

        # targets_ignored = 0
        # sources_ignored = 0

        for i in range(len(inputs)):
            s_mask = torch.zeros(inputs.shape[-2:],dtype=torch.bool).cuda()
            t_mask = torch.zeros(targets.shape[-2:],dtype=torch.bool).cuda()
            source_padding = data_samples[i].metainfo['padding_size']
            target_padding = target_data_samples[i].metainfo['padding_size']
            s_mask[:inputs.shape[-2:][0]-source_padding[-1],:inputs.shape[-2:][1]-source_padding[1]] = 1
            t_mask[:targets.shape[-2:][0]-target_padding[-1],:targets.shape[-2:][1]-target_padding[1]] = 1
            target_masks.append(t_mask)
            source_masks.append(s_mask)
        
        source_masks = torch.stack(source_masks,dim=0)
        target_masks = torch.stack(target_masks,dim=0)

        # print("targets shape", targets.shape)
        t_logits, t_p, t_attn = self.backbone.forward_features(targets)

        # print("inputs shape", inputs.shape)
        s_logits, s_p, s_attn = self.backbone.forward_features(inputs)
        
        loss_decode = self._decode_head_forward_train(s_logits, data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(s_logits, data_samples)
            losses.update(loss_aux)

        # t_logits = t_logits[-1:]
        # s_logits = s_logits[-1:]
        # t_attn = t_attn[-1:]
        # s_attn = s_attn[-1:]

        # print("Applying masks to source and target logits")

        for i in range(len(t_logits)):

            scale = int(target_masks.shape[-1]/t_logits[i].shape[-1])

            h_indexes = scale*torch.arange(target_masks.shape[-1]//scale,dtype=torch.int).cuda()
            v_indexes = scale*torch.arange(target_masks.shape[-2]//scale,dtype=torch.int).cuda()

            target_masks_scaled = torch.index_select(torch.index_select(target_masks,2,h_indexes),1,v_indexes).unsqueeze(1)
            source_masks_scaled = torch.index_select(torch.index_select(source_masks,2,h_indexes),1,v_indexes).unsqueeze(1)

            t_logits[i] = torch.mul(t_logits[i], target_masks_scaled)

            s_logits[i] = torch.mul(s_logits[i], source_masks_scaled)

        # print("Target predictions done")

        weight_tgt = torch.ones(self.num_classes).cuda()/self.num_classes

        weight_src = torch.ones(self.num_classes).cuda()/self.num_classes

        lambda_shape = (t_p.shape[0],*t_p.shape[-2:])

        t_lambda = dists.Beta(self.softplus(self.s_dist_alpha), self.softplus(self.s_dist_beta)).rsample(lambda_shape).squeeze(-1)

        s_lambda = 1-t_lambda

        #Returns dictionaries of mixup losses

        super_m_s_t_loss, unsuper_m_s_t_loss = self.mix_source_target(s_p,t_p,s_lambda,t_lambda,
                                                                      target_data_samples,data_samples,s_logits,
                                                                      t_logits,s_attn,t_attn,
                                                                      weight_tgt,weight_src,source_masks,
                                                                      target_masks)
        
        losses.update(super_m_s_t_loss)
        losses.update(unsuper_m_s_t_loss)

        return losses

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`], optional): The seg data
                samples. It usually includes information such as `metainfo`
                and `gt_sem_seg`.

        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        seg_logits = self.inference(inputs, batch_img_metas)

        return self.postprocess_result(seg_logits, data_samples)

    def forward(self,
                inputs: Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor',
                targets:Tensor = None,
                target_data_samples:OptSampleList = None) -> ForwardResults:

        """
        Args:
            inputs (torch.Tensor): The input tensor with shape (N, C, ...) in
                general.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`. Default to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`DetDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'loss':
            return self.loss(inputs, data_samples, targets, target_data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')
    
    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        x = self.extract_feat(inputs)
            
        return self.decode_head.forward(x)

    def slide_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                # change the image shape to patch shape
                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                # the output of encode_decode is seg logits tensor map
                # with shape [N, C, H, W]
                crop_seg_logit = self.encode_decode(crop_img, batch_img_metas)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        return seg_logits

    def whole_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference with full image.

        Args:
            inputs (Tensor): The tensor should have a shape NxCxHxW, which
                contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        seg_logits = self.encode_decode(inputs, batch_img_metas)

        return seg_logits

    def inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """Inference with slide/whole style.

        Args:
            inputs (Tensor): The input image of shape (N, 3, H, W).
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', 'pad_shape', and 'padding_size'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """
        assert self.test_cfg.get('mode', 'whole') in ['slide', 'whole'], \
            f'Only "slide" or "whole" test mode are supported, but got ' \
            f'{self.test_cfg["mode"]}.'
        ori_shape = batch_img_metas[0]['ori_shape']
        if not all(_['ori_shape'] == ori_shape for _ in batch_img_metas):
            print_log(
                'Image shapes are different in the batch.',
                logger='current',
                level=logging.WARN)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(inputs, batch_img_metas)
        else:
            seg_logit = self.whole_inference(inputs, batch_img_metas)

        return seg_logit

    def aug_test(self, inputs, batch_img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(inputs[0], batch_img_metas[0], rescale)
        for i in range(1, len(inputs)):
            cur_seg_logit = self.inference(inputs[i], batch_img_metas[i],
                                           rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(inputs)
        seg_pred = seg_logit.argmax(dim=1)
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
    
