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

# torch.autograd.set_detect_anomaly(True)


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

        self.s_dist_alpha = nn.Parameter(Tensor([1]),requires_grad=True)
        self.s_dist_beta = nn.Parameter(Tensor([1]),requires_grad=True)
        self.super_ratio = nn.Parameter(Tensor([-2]),requires_grad=False)
        self.unsuper_ratio = nn.Parameter(Tensor([-2]),requires_grad=False)

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
    
    # def mix_lambda_atten(self, s_scores, t_scores, s_lambda):
        
    #     #Returns a list of lambdas according to the number of layers of attention scores

    #     s_lambdas = []

    #     s_lambda = nn.Flatten()(s_lambda)
        
    #     for i in range(len(s_scores)):

    #         num_patch = s_scores[i].shape[-1] #'Source' patches in attention

    #         s_l = nn.AdaptiveAvgPool1d(num_patch)(s_lambda)
            
    #         t_l = 1-s_l

    #         s_lambda = torch.einsum('BLK,BK->BL',s_scores[i], s_l)
    #         t_lambda = torch.einsum('BLK,BK->BL',t_scores[i], t_l)
    #         s_lambdas.append(s_lambda/(s_lambda+t_lambda))

    #         # print(s_lambdas[i])

    #     return s_lambdas

    # def mix_lambda_atten(self, s_scores, t_scores, s_lambda):
        
    #     #Returns a list of lambdas according to the number of layers of attention scores

    #     s_lambda = nn.Flatten()(s_lambda)
        
    #     t_lambda = 1-s_lambda

    #     # s_lambda = torch.einsum('BLK,BK->BL',s_scores, s_lambda)
    #     # t_lambda = torch.einsum('BLK,BK->BL',t_scores, t_lambda)

    #     s_lambda = torch.mean(s_lambda,dim=-1)
    #     t_lambda = torch.mean(t_lambda,dim=-1)

    #     s_lambdas = []

    #     s_lambda = nn.Flatten()(s_lambda)
        
    #     for i in range(len(s_scores)):

    #         num_patch = s_scores[i].shape[-1] #'Source' patches in attention

    #         s_l = nn.AdaptiveAvgPool1d(num_patch)(s_lambda)
            
    #         t_l = 1-s_l

    #         s_lambda = torch.einsum('BLK,BK->BL',s_scores[i], s_l)
    #         t_lambda = torch.einsum('BLK,BK->BL',t_scores[i], t_l)
    #         s_lambdas.append(s_lambda/(s_lambda+t_lambda))

    #         # print(s_lambdas[i])

    #     return s_lambda/(s_lambda+t_lambda)

    def cosine_distance(self,s_logits, t_logits):
        # temp_matrix = []
        # for l in range(len(s_logits)):

        #     s_logit = nn.Flatten()(s_logits[l].permute(1,0,2,3))
        #     t_logit = nn.Flatten()(t_logits[l].permute(1,0,2,3))
        #     temp_matrix.append(torch.mm(s_logit.t(),t_logit))

        #     norm_s = torch.norm(s_logit,p=2,dim=0).unsqueeze(0)
        #     norm_t = torch.norm(t_logit,p=2,dim=0).unsqueeze(0)
        #     norm_matrix = torch.mm(norm_s.t(),norm_t)
        #     norm_matrix = torch.maximum(norm_matrix,torch.ones_like(norm_matrix).cuda()*1e-6)
        #     temp_matrix[l]/=norm_matrix

        s_logits = nn.Flatten()(s_logits.permute(1,0,2,3))
        t_logits = nn.Flatten()(t_logits.permute(1,0,2,3))

        # print(s_logit.shape,temp_matrix.shape)

        # s_logits_cropped = []

        # for batch in range(len(padding)):

        #     s_logits_cropped.append(nn.Flatten()(s_logits[batch,:,:s_logits.shape[2]-int(padding[batch][-1]/scale),:]))

        # s_logits = torch.cat(s_logits_cropped,dim=1)

        # print(s_logits.shape,t_logits.shape)

        temp_matrix = torch.mm(s_logits.t(),t_logits)

        norm_s = torch.norm(s_logits,p=2,dim=0).unsqueeze(0)
        norm_t = torch.norm(t_logits,p=2,dim=0).unsqueeze(0)
        norm_matrix = torch.mm(norm_s.t(),norm_t)
        norm_matrix = torch.maximum(norm_matrix,torch.ones_like(norm_matrix).cuda()*1e-6)
        temp_matrix/=norm_matrix

        # s_logit = nn.Flatten(start_dim=2)(s_logits) # B C (HW)
        # t_logit = nn.Flatten(start_dim=2)(t_logits) # B C (HW)
        # temp_matrix = torch.einsum('BCD, BCF -> BDF', s_logit, t_logit)

        # norm_s = torch.norm(s_logit,p=2,dim=1)
        # norm_t = torch.norm(t_logit,p=2,dim=1)
        # norm_matrix = torch.einsum('BN, BM -> BNM', norm_s, norm_t)
        # norm_matrix = torch.maximum(norm_matrix,torch.ones_like(norm_matrix).cuda()*1e-6)
        # temp_matrix/=norm_matrix
                
        return temp_matrix # B (HW) (HW)


    def mixup_supervised_dis(self, preds, s_label, lam, scale):

        # mixup_losses = torch.tensor(0.,requires_grad=True).cuda()

        # print(preds[0].shape)

        # print(preds[0].shape)

        # label_onehot = []

        label_onehot = F.one_hot(s_label,num_classes=256).permute(3,0,1,2)

        label_onehot = nn.Flatten()(nn.AvgPool2d(scale)(label_onehot.float()))

        # lams = []

        # for batch in range(len(padding)):
        #     # print(padding[batch])

        #     label_onehot.append(nn.Flatten()(s_label[batch,:,:s_label.shape[2]-int(padding[batch][-1]/scale),:]))

        #     lams.append(lam[0].repeat(1,label_onehot[-1].shape[-1]))

        # label_onehot = torch.cat(label_onehot,dim=1).float()

        # lams = torch.cat(lams,dim=1)

        # label_onehot = F.one_hot(s_label,num_classes=256)[:,:,:,:self.num_classes].permute(3,0,1,2)

        # print(s_label_tensor.shape)

        # label_onehot = nn.Flatten()(label_onehot).float()

        # print(slabel)

        # print(preds)

        # for p in range(len(preds)):

        # print(preds.shape, lams.shape)

        # label = nn.AvgPool1d(scale**2)(label_onehot)

        # lams = nn.AvgPool1d(scale**2)(lams)

        label_norm = torch.norm(label_onehot,p=2,dim=0).unsqueeze(0)

        norm_matrix = torch.mm(label_norm.t(),label_norm)

        norm_matrix = torch.maximum(norm_matrix, torch.ones_like(norm_matrix)*1e-6)

        slabel = torch.mm(label_onehot.t(),label_onehot).unsqueeze(0)

        slabel /= norm_matrix

        print("Source similiarity", slabel)
        print('intermediate-source',F.log_softmax(preds,dim=1))

        # lambda_ = nn.Flatten(start_dim=0)(lam[p]).unsqueeze(0)

        # print(slabel,slabel.shape)

        mixup_loss = -torch.sum(torch.mul(slabel.squeeze(0),F.log_softmax(preds,dim=1)),dim=1).unsqueeze(0)

        mixup_loss /= torch.sum(slabel,dim=1)

        # print('Mixup loss', mixup_loss)

        # print(torch.maximum(torch.sum(slabel,dim=1),torch.ones_like(mixup_loss)*1e-6))

        # print(mixup_loss)

        # mixup_losses += torch.mean(mixup_loss)

        # mixup_losses += torch.mm(lam[p],mixup_loss.t()).item()/lam[p].shape[-1]
        
        mixup_losses = torch.mean(torch.mul(lam,mixup_loss))
        # mixup_losses = torch.mean(mixup_loss)

        print('Mixup loss', mixup_losses)
        
        # print('Supervised',mixup_losses/len(preds))
        
        return mixup_losses

    # def mixup_supervised_dis(self, preds, s_label, lam):

    #     # print("shape of label", s_label.shape, preds.shape)

    #     # print('label', s_label)

    #     label_onehot = F.one_hot(s_label,num_classes=256)[:,:,:,:self.num_classes].permute(0,3,1,2)

    #     # print(label_onehot.shape)

    #     s_label = nn.AdaptiveAvgPool2d(int(preds.shape[-1]**0.5))(label_onehot.float())

    #     # print(s_label.shape)

    #     s_label = nn.Flatten(start_dim=2)(s_label)

    #     # print(s_label.shape)

    #     label_norm = torch.norm(s_label,p=2,dim=1)

    #     # print(label_norm)

    #     norm_matrix = torch.einsum('BM, BN -> BMN',label_norm,label_norm)

    #     norm_matrix = torch.maximum(norm_matrix, torch.ones_like(norm_matrix)*1e-6)

    #     slabel = torch.einsum('BCM, BCN -> BMN',s_label,s_label)

    #     slabel/=norm_matrix

    #     # print("supervised label similarity", slabel, preds)

    #     lambda_ = nn.Flatten(start_dim=1)(lam)

    #     mixup_loss = -torch.einsum('BMN, BMN -> BM', slabel, F.log_softmax(preds,dim=-1))

    #     mixup_loss /= torch.maximum(torch.sum(slabel,dim=-1),torch.ones_like(mixup_loss)*1e-6)

    #     mixup_loss = torch.mean(torch.mul(lambda_,mixup_loss))
        
    #     # print('Supervised',mixup_losses/len(preds))
        
    #     return mixup_loss # H x W


    def mixup_unsupervised_dis(self, preds, lam):

        # mixup_losses = torch.tensor(0.,requires_grad=True).cuda()
 
        # for p in range(len(preds)):

            # print(mask.shape)

        # mask_matrix = nn.Flatten()(mask.permute(1,0,2,3)).float()

        # mask_matrix = torch.mm(mask_matrix.t(),mask_matrix).bool()

        # print('mask_matrix',mask_matrix.shape, mask_matrix)

        # label = torch.mul(mask_matrix,torch.eye(preds.shape[0]).cuda())
        label = torch.eye(preds.shape[0]).cuda()

        # lams = []

        # for batch in range(len(padding)):
        #     # print(hw,int(padding[batch][-1]/scale))
        #     lams.append(lam[batch].repeat(1,(hw[0]-int(padding[batch][-1]/scale))*hw[1]))

        # # print(lams)

        # lams = torch.cat(lams,dim=1)

        # print(lams)

        # lams = nn.AvgPool1d(scale**2)(lams)
        
        # print(preds[p].shape)

        # print(-torch.sum(label*F.log_softmax(preds[p],dim=1)))

        # print('softmax',label*F.log_softmax(preds,dim=1))

        # mixup_losses += -torch.sum(label*F.log_softmax(preds[p],dim=1),dim=1).unsqueeze(0)
        # print(lams.shape,label.shape,preds.shape)
        mixup_losses = torch.mean(torch.mul(lam,-torch.sum(label*F.log_softmax(preds,dim=1),dim=1)))
        # preds = torch.maximum(preds,torch.ones_like(preds)*1e-6)
        # mixup_losses = 10*torch.mean(torch.mul(lam,-torch.sum(label*torch.log(preds),dim=1)))

        # lambda_ = nn.Flatten(start_dim=0)(lam[p]).unsqueeze(0)

        # mixup_losses += torch.mm(lambda_,mixup_loss.t()).item()/lambda_.shape[-1]
        
        # print('Unsupervised',mixup_losses/len(preds))

        return mixup_losses #N H x W
    
    # def mixup_unsupervised_dis(self, preds, lam):

    #     lam = nn.Flatten()(lam)

    #     label = []

    #     for batch in range(preds.shape[0]):

    #         label.append(torch.eye(preds.shape[1]).cuda())

    #     label = torch.stack(label,dim=0)

    #     # print(label.shape)

    #     mixup_loss = -torch.sum(label*F.log_softmax(preds,dim=-1),dim=-1)

    #     # print(mixup_loss.shape)

    #     mixup_loss = torch.mean(torch.mul(lam,mixup_loss))
        
    #     # print('Unsupervised',mixup_losses/len(preds))

    #     return mixup_loss # H x W
    
    def mixup_soft_ce(self, pred, labels, weight, lam):

        # pred_labels = []

        # for label in labels:
        #     pred_labels.append(label.gt_sem_seg.data.squeeze())
        
        # pred_labels = torch.stack(pred_labels)

        # print(lam.shape)

        # loss = torch.nn.CrossEntropyLoss(reduction='none',weight=weight,ignore_index=255)(pred, pred_labels)
        loss = lam*torch.nn.CrossEntropyLoss(reduction='mean',weight=weight,ignore_index=255)(pred, labels)
        # loss = torch.nn.CrossEntropyLoss(reduction='mean',weight=weight,ignore_index=255)(pred, labels)

        # print(loss)
        
        # count = torch.unique(pred_labels,return_counts=True)

        # if count[0][-1]==255:

        #     count[0][-1] = 0

        #     count[1][-1] = 0

        # count = torch.mm(weight[count[0]].unsqueeze(0),count[1].unsqueeze(-1).float()).item()

        # weight = torch.cat((weight,torch.zeros(256-self.num_classes).cuda()),0)
        
        # count = torch.sum(torch.mul(lam,weight[pred_labels]))

        # loss = torch.mean(lam)*(torch.sum(torch.mul(loss, lam))/count)

        return loss
    
    def mix_source_target(self, s_token, t_token, s_lambda, t_lambda,
                          target_label, infer_label, s_logits, t_logits,
                          s_scores, t_scores, weight_tgt, weight_src,
                        #   source_padding, target_padding
                          #source_masks, target_masks
                          ):
        
        # print("Mixing tokens")
        m_s_t_token = self.mix_token(s_token, t_token, s_lambda)

        # print("Forward Pass for intermediate domain")
        m_s_t_logits, m_s_t_p, _ = self.backbone.forward_features(m_s_t_token, p_in=True)

        # m_s_t_logits = m_s_t_logits[-1]
        
        # print("Generating intermediate domain predictions")
        m_s_t_pred = self.decode_head.forward(m_s_t_logits)

        m_s_t_pred = F.interpolate(m_s_t_pred,infer_label.shape[-2:],mode='bilinear',align_corners=True)

        # print('m_s_t_pred',m_s_t_pred)

        # m_s_t_pred = nn.Softmax(dim=1)(m_s_t_pred)

        m_s_t_logits = self.decode_head.forward(m_s_t_logits, logits=True)

        #Downscale

        # m_s_t_logits = m_s_t_logits[0]

        m_s_t_logits = nn.AvgPool2d(4)(m_s_t_logits)

        # scale = int(target_masks.shape[-1]/m_s_t_logits.shape[-1])

        # h_indexes = scale*torch.arange(target_masks.shape[-1]//scale,dtype=torch.int).cuda()
        # v_indexes = scale*torch.arange(target_masks.shape[-2]//scale,dtype=torch.int).cuda()

        # mask = torch.logical_or(source_masks,target_masks).cuda()

        # mask = torch.index_select(torch.index_select(mask,2,h_indexes),1,v_indexes)

        # print(mask, mask.shape)

        # m_s_t_logits = torch.mul(m_s_t_logits,mask)

        

        # m_s_t_logits = self.decode_head.forward(m_s_t_logits,logits=True)

        # print("Applying paddings masks on m_s_t_logits")

        # m_s_t_logits = m_s_t_logits[-1:]

        # for i in range(len(m_s_t_logits)):

        #     scale = int(target_masks.shape[-1]/m_s_t_logits[i].shape[-1])

        #     h_indexes = scale*torch.arange(target_masks.shape[-1]//scale,dtype=torch.int).cuda()
        #     v_indexes = scale*torch.arange(target_masks.shape[-2]//scale,dtype=torch.int).cuda()

        #     mask = torch.logical_or(source_masks,target_masks).cuda()

        #     mask = torch.index_select(torch.index_select(mask,2,h_indexes),1,v_indexes)

        #     # print(mask, mask.shape)

        #     m_s_t_logits[i] = torch.mul(m_s_t_logits[i],mask.unsqueeze(1))

        

        # print("Mixing Lambda Attention")
        # s_lambda = self.mix_lambda_atten(s_scores, t_scores, s_lambda) #B x HW

        hw = m_s_t_logits.shape[-1]*m_s_t_logits.shape[-2]

        s_lambda = torch.mean(nn.Flatten()(s_lambda),dim=1)

        s_lam = s_lambda

        # print('lambda',s_lam)

        s_lambda = s_lambda.repeat(1,hw).reshape(1,-1)

        # print(s_lambda)

        # print(s_lambda.shape)

        # print("s lambda", s_lambda)

        # print("lambda", s_lambda)

        # t_lambda = []

        # for i in range(len(s_lambda)):
        #     #Reshape s_lambda to layers x B x H x W
        #     lam_shape = (s_lambda[i].shape[0],*m_s_t_logits[i].shape[-2:])
        #     s_lambda[i] = torch.reshape(s_lambda[i],lam_shape)

        #     #Calculate t_lambda
        #     t_lambda.append(1-s_lambda[i])

        # lam_shape = (s_lambda.shape[0],*m_s_t_logits[i].shape[-2:])
        # s_lambda = torch.reshape(s_lambda,lam_shape)
        
        t_lambda = 1-s_lambda #B H W

        scale = infer_label.shape[-1]//m_s_t_logits.shape[-1]

        # print("Cosine Distance")
        # m_s_t_s = self.cosine_distance(m_s_t_logits, s_logits)

        # print('intermediate-source',m_s_t_s)

        # print("supervised", s_lambda, m_s_t_s, infer_label)

        # m_s_t_s_similarity = self.mixup_supervised_dis(m_s_t_s, infer_label, s_lambda, scale)

        # m_s_t_t = self.cosine_distance(m_s_t_logits,t_logits)

        # print('intermediate-target',m_s_t_t)

        # print('m_s_t_logits',m_s_t_logits)

        # print('t_logits',t_logits)


        # m_s_t_t_similarity = self.mixup_unsupervised_dis(m_s_t_t, t_lambda)

        # print("unsupervised", t_lambda, m_s_t_t)

        #resizing and concatenating feature_space_loss across layers to be same size as segmentation map

        # print('m_s_t_s_similarity', m_s_t_s_similarity)
        # print('m_s_t_t_similarity', m_s_t_t_similarity)

        # super_feature_space_loss = self.softplus(self.super_ratio)*m_s_t_s_similarity
        # unsuper_feature_space_loss = self.softplus(self.super_ratio)*m_s_t_t_similarity

        #resizing and concatenating lambda to be same size as segmentation map

        # length = len(s_lambda)
        
        # s_lam = F.interpolate(s_lambda[0].unsqueeze(0),size=infer_label[0].gt_sem_seg.data.shape[-2:],mode='bilinear',align_corners=True).squeeze(0)

        # for l in range(1,len(s_lambda)):
        #     s_lam += F.interpolate(s_lambda[l].unsqueeze(0),size=infer_label[0].gt_sem_seg.data.shape[-2:],mode='bilinear',align_corners=True).squeeze(0)
        
        # s_lam /= length

        # t_lam = 1 - s_lam

        # print("Mixup soft CE")
        super_m_s_t_s_loss = self.mixup_soft_ce(m_s_t_pred, infer_label, weight_src, s_lam)
        unsuper_m_s_t_loss = self.mixup_soft_ce(m_s_t_pred, target_label, weight_tgt, 1-s_lam)

        super_label_space_loss = self.softplus(self.unsuper_ratio)*super_m_s_t_s_loss
        unsuper_label_space_loss = self.softplus(self.unsuper_ratio)*unsuper_m_s_t_loss

        #Losses

        f_loss_dict = dict()
        l_loss_dict = dict()

        # f_losses = dict(loss_pm_feature=super_feature_space_loss, loss_pm_feature=unsuper_feature_space_loss)
        # l_losses = dict(loss_pm_label=label_space_loss)
        # l_losses = dict()
        # f_losses = dict()

        # f_loss_dict.update(add_prefix(dict(loss_pm_feature=super_feature_space_loss), 'super'))
        # f_loss_dict.update(add_prefix(dict(loss_pm_feature=unsuper_feature_space_loss), 'unsuper'))
        l_loss_dict.update(add_prefix(dict(loss_ce=super_label_space_loss), 'super'))
        l_loss_dict.update(add_prefix(dict(loss_ce=unsuper_label_space_loss), 'unsuper'))

        return f_loss_dict, l_loss_dict
    
    def attn_map(self, logits, attn, labels):

        weights = []

        channels = self.decode_head.channels

        num_patches = attn[0].shape[-1]

        labels = nn.Flatten(start_dim=1)(labels)

        # for idx in range(len(logits)):

        #     print(attn[idx].shape)

        # print(labels.shape)

        scale = int(labels.shape[-1]/num_patches)

        # print(scale)
        
        h_indexes = scale*torch.arange(num_patches,dtype=torch.int).cuda()

        labels = labels[:,h_indexes]

        for idx in range(len(logits)):
            
            weights.append(torch.einsum('BCHW, DC -> BDHW',logits[idx], self.decode_head.convs[idx].conv.weight.squeeze()))
            # print(weights[idx])
            weights[idx] = torch.einsum('BCHW, DC -> BDHW', weights[idx], self.decode_head.fusion_conv.conv.weight.squeeze()[:,idx*channels:(idx+1)*channels])
            # print(weights[idx])
            weights[idx] = torch.einsum('BCHW, DC -> BDHW', weights[idx], self.decode_head.conv_seg.weight.squeeze())
            # print(weights[idx])
            
            weights[idx] = nn.Flatten(start_dim=1,end_dim=2)(weights[idx].permute(0,2,3,1)) # B (HW) C

            weights[idx] = torch.cat([weights[idx],torch.zeros(*weights[idx].shape[:-1],256-weights[idx].shape[-1]).cuda()],dim=-1)

            # weights[idx] = F.interpolate(weights[idx].unsqueeze(1), num_patches, mode='bilinear', align_corners=True).squeeze(1)
            
            weights[idx] = torch.stack([weights[idx][batch][:,labels[batch]] for batch in range(len(labels))],dim=0) #B HW HW Batch x patches x labels


            # weights[idx] = weights[idx].gather(0,labels[h_indexes].unsqueeze(0)) #1 (BHW)

            weights[idx] = torch.einsum('BCN, BCD -> BND', attn[idx],weights[idx])

            # print(idx, weights[idx].shape)

            # print(weights[idx])

        # scores = torch.sum(torch.stack(weights,dim=0),dim=0)

        # print(scores.shape)

        scores = scores.softmax(dim=-1)

        # print(scores)

        # print(torch.max(scores,dim=1))

        # weights = self.decode_head.convs[0].conv.weight.squeeze()

        # print(weights.shape,weights)

        return scores #B x HW x HW

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

        weight_tgt: B*H*W x 1, confidence of label weight of each target label
        weight_src: B*H*W x 1, label weight of each source label

        mem_fea: x E, logits of all images pixels
        mem_cls: x C, probabilities of classification classes of all images pixels
        """

        #Create padding masks

        # print(self.s_dist_alpha, self.s_dist_beta, self.super_ratio, self.unsuper_ratio)

        # source_masks = []
        target_masks = []

        # source_padding = []
        # target_padding = []

        # targets_ignored = 0
        # sources_ignored = 0

        for i in range(len(inputs)):
            # s_mask = torch.zeros(inputs.shape[-2:],dtype=torch.bool).cuda()
            t_mask = torch.zeros(targets.shape[-2:],dtype=torch.bool).cuda()
            # source_padding.append(data_samples[i].metainfo['padding_size'])
            # target_padding.append(target_data_samples[i].metainfo['padding_size'])
            target_padding = target_data_samples[i].metainfo['padding_size']
            # s_mask[:inputs.shape[-2:][0]-source_padding[-1][-1],:inputs.shape[-2:][1]-source_padding[-1][1]] = 1
            t_mask[:targets.shape[-2:][0]-target_padding[-1],:targets.shape[-2:][1]-target_padding[1]] = 1
            target_masks.append(t_mask)
            # source_masks.append(s_mask)
        
        # source_masks = torch.stack(source_masks,dim=0)
        target_masks = torch.stack(target_masks,dim=0)

        # print("targets shape", targets.shape)
        t_logits, t_p, t_attn = self.backbone.forward_features(targets)

        # print("inputs shape", inputs.shape)
        s_logits, s_p, s_attn = self.backbone.forward_features(inputs)
        
        loss_decode = self._decode_head_forward_train(s_logits, data_samples)
        target_loss_decode = self._decode_head_forward_train(t_logits, target_data_samples)
        losses.update(loss_decode)
        losses.update(dict(target_l=target_loss_decode['decode.loss_ce']))

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(s_logits, data_samples)
            losses.update(loss_aux)

        #Flatten and interpolate attentions

        # num_patch = int(s_attn[0].shape[1]**0.5)

        # for idx in range(len(s_attn)):

        #     hw = int(s_attn[idx].shape[-1]**0.5)

        #     # print(idx)

        #     # print("attention before normalisation", s_attn[idx])

        #     s_attn[idx] = F.interpolate(s_attn[idx].reshape(*s_attn[idx].shape[:2],hw,hw),num_patch,mode='bilinear',align_corners=True)
        #     t_attn[idx] = F.interpolate(t_attn[idx].reshape(*t_attn[idx].shape[:2],hw,hw),num_patch,mode='bilinear',align_corners=True)

        #     s_attn[idx] = nn.Flatten(start_dim=2)(s_attn[idx])
        #     t_attn[idx] = nn.Flatten(start_dim=2)(t_attn[idx])

        #     s_norm = torch.sum(s_attn[idx],dim=-1).unsqueeze(-1)
        #     t_norm = torch.sum(t_attn[idx],dim=-1).unsqueeze(-1)

        #     s_attn[idx]/=s_norm
        #     t_attn[idx]/=t_norm

        #     if idx: 
        #         s_attn[idx] = torch.einsum('BNC, BCM -> BNM',s_attn[idx],s_attn[idx-1])
        #         t_attn[idx] = torch.einsum('BNC, BCM -> BNM',t_attn[idx],t_attn[idx-1])
            
        #     # print(s_attn[idx].shape, s_norm.shape, s_norm)

        #     num_patch = int(s_attn[idx].shape[-2]**0.5)

            # print(num_patch)
            # print("attention scores after scaling and normalisation")
            # print(s_attn[idx])
            # print(torch.sum(s_attn[idx],-1))

        #Pseudo labelling

        # t_labels = self.decode_head.forward(t_logits)

        # t_labels = F.interpolate(t_labels,data_samples[0].gt_sem_seg.data.shape[-2:],mode='bilinear',align_corners=True)

        # t_labels = torch.argmax(nn.Softmax(dim=1)(t_labels),dim=1).squeeze(1)

        # t_labels = torch.mul(t_labels,target_masks) + 255*torch.logical_not(target_masks)

        # print(t_labels.shape)
        
        #Format labels

        s_labels = []

        for label in data_samples:

            s_labels.append(label.gt_sem_seg.data.squeeze())
        
        s_labels = torch.stack(s_labels,dim=0).cuda()

        t_labels = []

        for label in target_data_samples:

            t_labels.append(label.gt_sem_seg.data.squeeze())
        
        t_labels = torch.stack(t_labels,dim=0).cuda()

        # print("Applying masks to source and target logits")

        # s_logits = self.decode_head.forward(s_logits, logits=True)

        # t_logits = self.decode_head.forward(t_logits, logits=True)

        # s_logits = s_logits[0]
        # t_logits = t_logits[0]

        # for i in range(len(t_logits)):

        #     scale = int(target_masks.shape[-1]/t_logits[i].shape[-1])

        #     h_indexes = scale*torch.arange(target_masks.shape[-1]//scale,dtype=torch.int).cuda()
        #     v_indexes = scale*torch.arange(target_masks.shape[-2]//scale,dtype=torch.int).cuda()

        #     target_masks_scaled = torch.index_select(torch.index_select(target_masks,2,h_indexes),1,v_indexes).unsqueeze(1)
        #     source_masks_scaled = torch.index_select(torch.index_select(source_masks,2,h_indexes),1,v_indexes).unsqueeze(1)

        #     t_logits[i] = torch.mul(t_logits[i], target_masks_scaled)

        #     s_logits[i] = torch.mul(s_logits[i], source_masks_scaled)

        # s_logits = nn.AvgPool2d(4)(s_logits)
        # t_logits = nn.AvgPool2d(4)(t_logits)

        # s_logits_cropped = []
        # t_logits_cropped = []

        # scale = s_labels.shape[-1]/s_logits.shape[-1]

        # for batch in range(len(source_padding)):

        #     # print(source_padding[batch])
        #     # print(target_padding[batch])

        #     s_logits_cropped.append(nn.Flatten()(s_logits[batch,:,:s_logits.shape[2]-int(source_padding[batch][-1]/scale),:]))

        #     t_logits_cropped.append(nn.Flatten()(t_logits[batch,:,:t_logits.shape[2]-int(target_padding[batch][-1]/scale),:]))

        # s_logits = torch.cat(s_logits_cropped,dim=1)
        # t_logits = torch.cat(t_logits_cropped,dim=1)

        # print('s_logits shape',s_logits.shape)
        # print('t_logits shape',t_logits.shape)

        # scale = int(target_masks.shape[-1]/t_logits.shape[-1])

        # h_indexes = scale*torch.arange(target_masks.shape[-1]//scale,dtype=torch.int).cuda()
        # v_indexes = scale*torch.arange(target_masks.shape[-2]//scale,dtype=torch.int).cuda()

        # target_masks_scaled = torch.index_select(torch.index_select(target_masks,2,h_indexes),1,v_indexes).unsqueeze(1)
        # source_masks_scaled = torch.index_select(torch.index_select(source_masks,2,h_indexes),1,v_indexes).unsqueeze(1)
        
        # print('target_mask',target_masks_scaled.shape,target_masks_scaled)

        # print('t_logits before mask', t_logits)

        # t_logits = torch.mul(t_logits, target_masks_scaled)

        # s_logits = torch.mul(s_logits, source_masks_scaled)

        # print('t_logits',t_logits.shape,t_logits)
        # print('s_logits',s_logits.shape,s_logits)

        #Generating attention scores

        # s_scores = self.attn_map(s_logits, s_attn, s_labels)

        # t_scores = self.attn_map(t_logits, t_attn, t_labels)
        

        # t_logits = self.decode_head.forward(t_logits,logits=True)

        # s_logits = self.decode_head.forward(s_logits,logits=True)

        # t_logits = t_logits[-1:]
        # s_logits = s_logits[-1:]
        # t_attn = t_attn[-1:]
        # s_attn = s_attn[-1:]

        # scale = int(target_masks.shape[-1]/t_logits.shape[-1])

        # h_indexes = scale*torch.arange(target_masks.shape[-1]//scale,dtype=torch.int).cuda()
        # v_indexes = scale*torch.arange(target_masks.shape[-2]//scale,dtype=torch.int).cuda()

        # target_masks_scaled = torch.index_select(torch.index_select(target_masks,2,h_indexes),1,v_indexes).unsqueeze(1)
        # source_masks_scaled = torch.index_select(torch.index_select(source_masks,2,h_indexes),1,v_indexes).unsqueeze(1)

        # t_logits = torch.mul(t_logits, target_masks_scaled)

        # s_logits = torch.mul(s_logits, source_masks_scaled)

        # print("Target predictions done")

        weight_tgt = torch.ones(self.num_classes).cuda()/self.num_classes

        weight_src = torch.ones(self.num_classes).cuda()/self.num_classes

        stride = 1

        lambda_shape = (t_p.shape[0],t_p.shape[-2]//stride,t_p.shape[-1]//stride)

        t_lambda = dists.Beta(self.softplus(self.s_dist_alpha), self.softplus(self.s_dist_beta)).rsample(lambda_shape).squeeze(-1)

        # print(t_lambda)

        # t_lambda = F.interpolate(t_lambda.unsqueeze(1),t_p.shape[-2:],mode='bilinear',align_corners=True).squeeze(1)
        # t_lambda = F.interpolate(t_lambda.unsqueeze(1),targets.shape[-2:]).squeeze(1)

        # print('interpolated',t_lambda)

        s_lambda = 1-t_lambda

        # print('label similarity', torch.mean((s_labels==t_labels).float()))

        # print(self.super_ratio,self.softplus(self.super_ratio))
        # print(self.unsuper_ratio,self.softplus(self.unsuper_ratio))

        #Returns dictionaries of mixup losses

        super_m_s_t_loss, unsuper_m_s_t_loss = self.mix_source_target(s_p,t_p,s_lambda,t_lambda,
                                                                      t_labels,s_labels,s_logits,
                                                                      t_logits,s_attn,t_attn,
                                                                      weight_tgt,weight_src)
        
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
    
    
    
