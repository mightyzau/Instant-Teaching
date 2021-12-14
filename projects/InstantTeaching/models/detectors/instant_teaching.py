#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
import numpy as np
from copy import deepcopy

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler, bbox_overlaps, multiclass_nms
from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck
from mmdet.models.detectors.base import BaseDetector


@DETECTORS.register_module()
class InstantTeachingTwoStageDetector(BaseDetector):
    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 ssl_warmup_iters=-1,
                 ssl_gt_box_low_thr=-1,                                
                 ssl_with_mixup=False, ssl_with_mosaic=False,
                 ssl_with_mixup_labeled=False, ssl_with_mosaic_labeled=False,
                 ssl_with_co_rectify=False,
                 ssl_lambda_u=1.0,                                       # loss weight for unlabeled images
        ):
        self.ssl_warmup_iters = ssl_warmup_iters
        self.iter = 0
        self.ssl_gt_box_low_thr = ssl_gt_box_low_thr

        self.ssl_with_mixup = ssl_with_mixup             
        self.ssl_with_mosaic = ssl_with_mosaic
        self.ssl_with_mixup_labeled = ssl_with_mixup_labeled
        self.ssl_with_mosaic_labeled = ssl_with_mosaic_labeled
        self.ssl_lambda_u = ssl_lambda_u

        self.ssl_with_co_rectify = ssl_with_co_rectify

        super(InstantTeachingTwoStageDetector, self).__init__()
        self.backbone = build_backbone(backbone)
        if self.ssl_with_co_rectify:
            self.backbone2 = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)
            if self.ssl_with_co_rectify:
                self.neck2 = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)
            if self.ssl_with_co_rectify:
                self.rpn_head2 = build_head(rpn_head_)

        if roi_head is not None:
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = build_head(roi_head)
            if self.ssl_with_co_rectify:
                self.roi_head2 = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)
    
    def train_step(self, data, optimizer):
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))
        self.iter += 1
        return outputs

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None
    
    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self.roi_head, 'bbox_head')
                 and self.roi_head.bbox_head is not None)
                or (hasattr(self, 'bbox_head') and self.bbox_head is not None))

    def init_weights(self, pretrained=None):
        super(InstantTeachingTwoStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_roi_head:
            self.roi_head.init_weights(pretrained)
            
        if self.ssl_with_co_rectify:
            self.backbone2.init_weights(pretrained=pretrained)
            if self.with_neck:
                if isinstance(self.neck2, nn.Sequential):
                    for m in self.neck2:
                        m.init_weights()
                else:
                    self.neck2.init_weights()
            if self.with_rpn:
                self.rpn_head2.init_weights()
            if self.with_roi_head:
                self.roi_head2.init_weights(pretrained)


    def extract_feat(self, img, idx=1):
        """Directly extract features from the backbone+neck
        """
        assert idx in [1, 2]
        if idx == 2:
            assert self.ssl_with_co_rectify

        if idx == 2:
            x = self.backbone2(img)
            if self.with_neck:
                x = self.neck2(x)
        else:
            x = self.backbone(img)
            if self.with_neck:
                x = self.neck(x)
        return x

    
    def forward_dummy(self, img):
        outs = ()
        x = self.extract_feat(img)
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def _net_pred(self, x, img_metas, rpn_head, roi_head):
        proposal_list = rpn_head.simple_test_rpn(x, img_metas)
        num_per_im = [len(p) for p in proposal_list]
        rois = bbox2roi(proposal_list)
        bbox_results = roi_head._bbox_forward(x, rois)
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        
        if not roi_head.bbox_head.use_sigmoid:
            scores = F.softmax(cls_score, dim=1)
        else:
            scores = cls_score.sigmoid()
        bboxes = roi_head.bbox_head.bbox_coder.decode(rois[:, 1:], bbox_pred)
        if bboxes.size(1) > 4:
            bboxes = bboxes.view(bboxes.size(0), -1, 4)
            bboxes = bboxes[torch.arange(bboxes.size(0)), scores[:, :-1].argmax(1), :]
        return torch.split(scores, num_per_im), torch.split(bboxes, num_per_im)

    def _inference_single(self, x_u, 
                            img_metas_u, 
                            rpn_head, 
                            roi_head, 
                            rcnn_test_cfg, 
                            rescale=True,                                                
        ):
        """ compute scores and bbox prediction for a single image
        """
        num_classes = roi_head.bbox_head.num_classes
        assert x_u[0].size(0) == 1
        assert len(img_metas_u) == 1
        scores_p, bboxes_p = self._net_pred(x_u, img_metas_u, rpn_head, roi_head)
        assert len(scores_p) == 1
        assert len(bboxes_p) == 1

        cls_p = scores_p[0]
        box_p = bboxes_p[0]

        score_thr = 0.5
        if self.ssl_gt_box_low_thr > 0:
            score_thr = self.ssl_gt_box_low_thr

        keep = cls_p[:, :-1].max(1)[0] >= score_thr
        cls_p = cls_p[keep]
        box_p = box_p[keep]
        if box_p.numel() > 0:
            assert box_p.size(1) == 4
            if rescale:
                # data_aug: resize -> flip -> pad
                if img_metas_u[0]['flip']:
                    _h, _w, _ = img_metas_u[0]['img_shape']
                    box_p[:, [0, 2]] = _w - box_p[:, [2, 0]]
                box_p = box_p / box_p.new_tensor(img_metas_u[0]['scale_factor']).reshape(1, 4)

        return cls_p, box_p

    def _update_oneline(self, img_w, img_metas_w, gt_bboxes_w, gt_labels_w,
                         img_metas_s, gt_bboxes_s, gt_labels_s, gt_scores_s,
                         unlabeled_img_ids, rpn_head, roi_head, idx=1):
        rcnn_test_cfg = roi_head.test_cfg
        num_classes = roi_head.bbox_head.num_classes
        for i in unlabeled_img_ids:
            with torch.no_grad():
                ## inference online pseudo labels and transform to original image size
                _x_i = self.extract_feat(img_w[[i]], idx=idx)
                _p_score, _p_bbox = self._inference_single(_x_i, [img_metas_w[i]], rpn_head, 
                                                roi_head, rcnn_test_cfg, rescale=True)
                
                ## convert offline pseudo labels to originla image size
                _gt_bbox = gt_bboxes_w[i]
                if img_metas_w[i]['flip']:
                    _h, _w, _ = img_metas_w[i]['img_shape']
                    assert _gt_bbox.size(1) == 4
                    _gt_bbox[:, [0, 2]] = _w - _gt_bbox[:, [2, 0]]
                _gt_bbox = _gt_bbox / _gt_bbox.new_tensor(img_metas_w[i]['scale_factor']).reshape(1, 4)
                _gt_score = gt_labels_w[i]

                ## nms
                _dets, _labels = multiclass_nms(
                                    torch.cat((_p_bbox, _gt_bbox)), 
                                    torch.cat((_p_score, _gt_score)),
                                    rcnn_test_cfg.score_thr, rcnn_test_cfg.nms,
                                    rcnn_test_cfg.max_per_img, num_classes=num_classes)
                _bboxes = _dets[:, :4]
                _scores = _dets[:, -1]
                
                ## transform to strong aug: resize -> flip -> pad
                _bboxes = _bboxes * _bboxes.new_tensor(img_metas_s[i]['scale_factor']).reshape(-1, 4)
                if img_metas_s[i]['flip']:
                    _h, _w, _ = img_metas_s[i]['img_shape']
                    assert _bboxes.size(1) == 4
                    _bboxes[:, [0, 2]] = _w - _bboxes[:, [2, 0]]

                gt_bboxes_s[i] = _bboxes
                gt_labels_s[i] = _labels
                gt_scores_s[i] = _scores
                
        return gt_bboxes_s, gt_labels_s, gt_scores_s
    
    def _roi_head_pred(self, x, roi_head, proposal_list, img_meta):
        _h, _w, _ = img_meta['img_shape']
        assert x[0].size(0) == 1
        assert len(proposal_list) == 1
        num_per_im = [len(p) for p in proposal_list]
        rois = bbox2roi(proposal_list)
        bbox_results = roi_head._bbox_forward(x, rois)
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']

        if not roi_head.bbox_head.use_sigmoid:
            scores = F.softmax(cls_score, dim=1)
        else:
            scores = cls_score.sigmoid()
        bboxes = roi_head.bbox_head.bbox_coder.decode(rois[:, 1:], bbox_pred)
        if bboxes.size(1) > 4:
            bboxes = bboxes.view(bboxes.size(0), -1, 4)
            bboxes = bboxes[torch.arange(bboxes.size(0)), scores[:, :-1].argmax(1), :]
        bboxes[..., [0, 2]] = bboxes[..., [0, 2]].clamp(0, _w)
        bboxes[..., [1, 3]] = bboxes[..., [1, 3]].clamp(0, _h)

        scores, bboxes = torch.split(scores, num_per_im), torch.split(bboxes, num_per_im)
        assert len(scores) == 1
        assert len(bboxes) == 1
        return scores[0], bboxes[0]

    def _update_oneline_co_rectify(self, img_w, img_metas_w, gt_bboxes_w, gt_labels_w,
                         img_metas_s, gt_bboxes_s, gt_labels_s, gt_scores_s,
                         img_metas_s2, gt_bboxes_s2, gt_labels_s2, gt_scores_s2,
                         unlabeled_img_ids, rpn_head, roi_head, rpn_head2, roi_head2):

        rcnn_test_cfg = roi_head.test_cfg
        num_classes = roi_head.bbox_head.num_classes
        for i in unlabeled_img_ids:
            with torch.no_grad():
                assert gt_labels_w[i].max() < 0.5           ## no offline pseudo annotations

                _x_i_1 = self.extract_feat(img_w[[i]], idx=1)
                _scores_1, _bboxes_1 = self._inference_single(_x_i_1, [img_metas_w[i]], rpn_head, 
                                                roi_head, rcnn_test_cfg, rescale=False)
                _dets_1, _labels_1 = multiclass_nms(_bboxes_1, _scores_1,
                                    rcnn_test_cfg.score_thr, rcnn_test_cfg.nms,
                                    rcnn_test_cfg.max_per_img, num_classes=num_classes)
                _bboxes_1 = _dets_1[:, :4]
                _scores_1 = _dets_1[:, -1]
                    
                _x_i_2 = self.extract_feat(img_w[[i]], idx=2)
                _scores_2, _bboxes_2 = self._inference_single(_x_i_2, [img_metas_w[i]], rpn_head2, 
                                                roi_head2, rcnn_test_cfg, rescale=False)
                _dets_2, _labels_2 = multiclass_nms(_bboxes_2, _scores_2,
                                    rcnn_test_cfg.score_thr, rcnn_test_cfg.nms,
                                    rcnn_test_cfg.max_per_img, num_classes=num_classes)
                _bboxes_2 = _dets_2[:, :4]
                _scores_2 = _dets_2[:, -1]

                _co_bboxes = torch.cat((_bboxes_1, _bboxes_2))
                _co_scores = torch.cat((_scores_1, _scores_2))
                _co_labels = torch.cat((_labels_1, _labels_2))
                
  
                # first step: use head_1 and head_2 to generate co pseudo annotations a
                #  second step: use head_1 and head_2 to rectify respectively, and concat a for annotations a_r
                if len(_co_bboxes) > 0:
                    _co_scores = _co_scores.new_zeros(_co_scores.size(0), num_classes + 1, dtype=torch.float32).scatter_(1, _co_labels.view(-1, 1), _co_scores.view(-1, 1))
                    _scores_1_r, _bboxes_1_r = self._roi_head_pred(_x_i_1, self.roi_head, [_co_bboxes], img_metas_w[i])
                    _bboxes_1_r = (_co_bboxes * _co_scores.max(1)[0][:, None] + _bboxes_1_r * _scores_1_r.max(1)[0][:, None]) / (_co_scores.max(1)[0][:, None] + _scores_1_r.max(1)[0][:, None])
                    _scores_1_r = (_scores_1_r + _co_scores) / 2

                    _scores_2_r, _bboxes_2_r = self._roi_head_pred(_x_i_2, self.roi_head2, [_co_bboxes], img_metas_w[i])
                    _bboxes_2_r = (_co_bboxes * _co_scores.max(1)[0][:, None] + _bboxes_2_r * _scores_2_r.max(1)[0][:, None]) / (_co_scores.max(1)[0][:, None] + _scores_2_r.max(1)[0][:, None])
                    _scores_2_r = (_scores_2_r + _co_scores) / 2

                    _co_bboxes_new = torch.cat((_bboxes_1_r, _bboxes_2_r, _co_bboxes))
                    _co_scores_new = torch.cat((_scores_1_r, _scores_2_r, _co_scores))
                    
                    # pseudo annotations for head_1
                    _scores_1_r, _bboxes_1_r = self._roi_head_pred(_x_i_2, self.roi_head2, [_co_bboxes_new], img_metas_w[i])
                    _scores_1_r = (_scores_1_r + _co_scores_new) / 2
                    _bboxes_1_r = (_co_bboxes_new * _co_scores_new.max(1)[0][:, None] + _bboxes_1_r * _scores_1_r.max(1)[0][:, None]) / (_co_scores_new.max(1)[0][:, None] + _scores_1_r.max(1)[0][:, None])
                    _dets_1_r, _labels_1_r = multiclass_nms(_bboxes_1_r, _scores_1_r,
                                        rcnn_test_cfg.score_thr, rcnn_test_cfg.nms,
                                        rcnn_test_cfg.max_per_img, num_classes=num_classes)
                    _bboxes_1_r = _dets_1_r[:, :4]
                    _scores_1_r = _dets_1_r[:, -1]

                    # pseudo annotations for head_2
                    _scores_2_r, _bboxes_2_r = self._roi_head_pred(_x_i_1, self.roi_head, [_co_bboxes_new], img_metas_w[i])
                    _scores_2_r = (_scores_2_r + _co_scores_new) / 2
                    _bboxes_2_r = (_co_bboxes_new * _co_scores_new.max(1)[0][:, None] + _bboxes_2_r * _scores_2_r.max(1)[0][:, None]) / (_co_scores_new.max(1)[0][:, None] + _scores_2_r.max(1)[0][:, None])
                    _dets_2_r, _labels_2_r = multiclass_nms(_bboxes_2_r, _scores_2_r,
                                        rcnn_test_cfg.score_thr, rcnn_test_cfg.nms,
                                        rcnn_test_cfg.max_per_img, num_classes=num_classes)
                    _bboxes_2_r = _dets_2_r[:, :4]
                    _scores_2_r = _dets_2_r[:, -1]
                else:
                    _bboxes_1_r, _scores_1_r, _labels_1_r = _co_bboxes, _co_scores, _co_labels
                    _bboxes_2_r, _scores_2_r, _labels_2_r = _co_bboxes, _co_scores, _co_labels

   
                ## convert to originla image size
                if img_metas_w[i]['flip']:
                    _h, _w, _ = img_metas_w[i]['img_shape']
                    assert _bboxes_1_r.size(1) == 4
                    assert _bboxes_2_r.size(1) == 4
                    _bboxes_1_r[:, [0, 2]] = _w - _bboxes_1_r[:, [2, 0]]
                    _bboxes_2_r[:, [0, 2]] = _w - _bboxes_2_r[:, [2, 0]]
                _bboxes_1_r = _bboxes_1_r / _bboxes_1_r.new_tensor(img_metas_w[i]['scale_factor']).reshape(1, 4)
                _bboxes_2_r = _bboxes_2_r / _bboxes_2_r.new_tensor(img_metas_w[i]['scale_factor']).reshape(1, 4)

                ## transform to strong aug: resize -> flip -> pad
                _bboxes_1_r = _bboxes_1_r * _bboxes_1_r.new_tensor(img_metas_s[i]['scale_factor']).reshape(-1, 4)
                if img_metas_s[i]['flip']:
                    _h, _w, _ = img_metas_s[i]['img_shape']
                    _bboxes_1_r[:, [0, 2]] = _w - _bboxes_1_r[:, [2, 0]]
                gt_bboxes_s[i] = _bboxes_1_r
                gt_labels_s[i] = _labels_1_r
                gt_scores_s[i] = _scores_1_r

                ## transform to strong aug: resize -> flip -> pad
                _bboxes_2_r = _bboxes_2_r * _bboxes_2_r.new_tensor(img_metas_s2[i]['scale_factor']).reshape(-1, 4)
                if img_metas_s2[i]['flip']:
                    _h, _w, _ = img_metas_s2[i]['img_shape']
                    _bboxes_2_r[:, [0, 2]] = _w - _bboxes_2_r[:, [2, 0]]
                gt_bboxes_s2[i] = _bboxes_2_r
                gt_labels_s2[i] = _labels_2_r
                gt_scores_s2[i] = _scores_2_r
                
        return gt_bboxes_s, gt_labels_s, gt_scores_s, gt_bboxes_s2, gt_labels_s2, gt_scores_s2

    
    def _map_gt_infos(self, gt_infos):
        #gt_infos_l_w, gt_infos_l_s, gt_infos_l_s_2, gt_infos_u_w, gt_infos_u_s, gt_infos_u_s_2 = gt_infos
        #return (gt_infos_l_w + gt_infos_u_w, gt_infos_l_s + gt_infos_u_s, gt_infos_l_s_2 + gt_infos_u_s_2)
        gt_infos_w, gt_infos_s, gt_infos_s_2 = [], [], []
        for s in range(0, len(gt_infos), 3):
            gt_infos_w += gt_infos[s]
            gt_infos_s += gt_infos[s+1]
            gt_infos_s_2 += gt_infos[s+2]
        return (gt_infos_w, gt_infos_s, gt_infos_s_2)


    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_scores=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        assert proposals is None and gt_masks is None
        assert gt_scores is not None
        num_classes = self.roi_head.bbox_head.num_classes


        ## 解析 images: each image has three aug in sequense: [weak_aug, s_aug, s_aug_2]
        ##      img_metas: [(w_l, s_l, s2_l, w_u, s_u, s2_u), ...] length of batch (label or unlabel)
        ##      img:   [b_w_l, b_s_l, b_s2_l, b_w_u, b_s_u, b_s2_u]     
        assert len(img) == 6
        img_metas_w = [{k: m[k][0] for k in m} for m in img_metas] + [{k: m[k][3] for k in m} for m in img_metas]
        img_metas_s = [{k: m[k][1] for k in m} for m in img_metas] +  [{k: m[k][4] for k in m} for m in img_metas]
        img_metas_s_2 = [{k: m[k][2] for k in m} for m in img_metas] + [{k: m[k][5] for k in m} for m in img_metas]

        img_w = img[0].new_zeros((img[0].size(0)*2, 3, max(img[0].size(2), img[3].size(2)), max(img[0].size(3), img[3].size(3))))
        img_w[:img[0].size(0), :, :img[0].size(2), :img[0].size(3)] = img[0]
        img_w[-img[3].size(0):, :, :img[3].size(2), :img[3].size(3)] = img[3]
        for i in range(len(img_metas_w)):
            img_metas_w[i]['pad_shape'] = (img_w.size(2), img_w.size(3), 3)

        img_s = img[1].new_zeros((img[1].size(0)*2, 3, max(img[1].size(2), img[4].size(2)), max(img[1].size(3), img[4].size(3))))
        img_s[:img[1].size(0), :, :img[1].size(2), :img[1].size(3)] = img[1]
        img_s[-img[4].size(0):, :, :img[4].size(2), :img[4].size(3)] = img[4]
        for i in range(len(img_metas_s)):
            img_metas_s[i]['pad_shape'] = (img_s.size(2), img_s.size(3), 3)

        img_s_2 = img[2].new_zeros((img[2].size(0)*2, 3, max(img[2].size(2), img[5].size(2)), max(img[2].size(3), img[5].size(3))))
        img_s_2[:img[2].size(0), :, :img[2].size(2), :img[2].size(3)] = img[2]
        img_s_2[-img[5].size(0):, :, :img[5].size(2), :img[5].size(3)] = img[5]
        for i in range(len(img_metas_s_2)):
            img_metas_s_2[i]['pad_shape'] = (img_s_2.size(2), img_s_2.size(3), 3)

        gt_bboxes_w, gt_bboxes_s, gt_bboxes_s_2 = self._map_gt_infos(gt_bboxes)                     # list[(n, 4), ...]
        gt_labels_w, gt_labels_s, gt_labels_s_2 = self._map_gt_infos(gt_labels)                     # list[(n,), ...]
        gt_scores_w, gt_scores_s, gt_scores_s_2 = self._map_gt_infos(gt_scores)  

        assert gt_bboxes_ignore is None
        gt_bboxes_ignore_s = None
        gt_bboxes_ignore_s_2 = None

        ## filter out is_unlabeled images
        unlabeled_img_ids = []
        labeled_img_ids = []
        for i, (img_info_w, img_info_s, img_info_s_2) in enumerate(zip(img_metas_w, img_metas_s, img_metas_s_2)):
            assert img_info_w['filename'] == img_info_s['filename'] == img_info_s_2['filename']
            assert img_info_w['is_unlabeled'] == img_info_s['is_unlabeled'] == img_info_s['is_unlabeled']
            if img_info_w['is_unlabeled']:
                unlabeled_img_ids.append(i)
                assert not torch.all(gt_scores_s[i] >= 1.0)
                assert not torch.all(gt_scores_s_2[i] >= 1.0)
            else:
                labeled_img_ids.append(i)
                assert torch.all(gt_scores_s[i] >= 1.0)
                assert torch.all(gt_scores_s_2[i] >= 1.0)

        if self.ssl_gt_box_low_thr > 0:
            num_pos_pre = sum([(s >= self.ssl_gt_box_low_thr).sum().item() for s in gt_scores_s])
            num_pos_pre2 = sum([(s >= self.ssl_gt_box_low_thr).sum().item() for s in gt_scores_s_2])
        else:
            num_pos_pre = sum([len(s) for s in gt_scores_s])
            num_pos_pre2 = sum([len(s) for s in gt_scores_s_2])

        ## convert gt_labels to soft labels with scores
        gt_labels_w = [g.new_zeros(g.size(0), num_classes + 1, dtype=torch.float32).scatter_(1, g.view(-1, 1), s.view(-1, 1))
                        for (g, s) in zip(gt_labels_w, gt_scores_w) ]
        gt_labels_s = [g.new_zeros(g.size(0), num_classes + 1, dtype=torch.float32).scatter_(1, g.view(-1, 1), s.view(-1, 1))
                        for (g, s) in zip(gt_labels_s, gt_scores_s)]
        gt_labels_s_2 = [g.new_zeros(g.size(0), num_classes + 1, dtype=torch.float32).scatter_(1, g.view(-1, 1), s.view(-1, 1))
                        for (g, s) in zip(gt_labels_s_2, gt_scores_s_2)]

        if (self.iter > self.ssl_warmup_iters):
            ## prediction on weak_aug_unlabeled images, and then update strong_aug_unlabeled groundtruth
            if self.ssl_with_co_rectify:
                gt_bboxes_s, gt_labels_s, gt_scores_s, gt_bboxes_s_2, gt_labels_s_2, gt_scores_s_2 = self._update_oneline_co_rectify(img_w, img_metas_w, gt_bboxes_w, gt_labels_w,
                            img_metas_s, gt_bboxes_s, gt_labels_s, gt_scores_s,
                            img_metas_s_2, gt_bboxes_s_2, gt_labels_s_2, gt_scores_s_2,
                            unlabeled_img_ids, self.rpn_head, self.roi_head, self.rpn_head2, self.roi_head2)
            else:
                gt_bboxes_s, gt_labels_s, gt_scores_s = self._update_oneline(img_w, img_metas_w, gt_bboxes_w, gt_labels_w,
                            img_metas_s, gt_bboxes_s, gt_labels_s, gt_scores_s,
                            unlabeled_img_ids, self.rpn_head, self.roi_head, idx=1)

        if self.ssl_gt_box_low_thr > 0:
            for b in range(len(gt_labels_s)):
                assert gt_labels_s[b].shape[0] == gt_scores_s[b].shape[0]
                keep_idx_s = gt_scores_s[b] >= self.ssl_gt_box_low_thr
                gt_bboxes_s[b] = gt_bboxes_s[b][keep_idx_s]
                gt_labels_s[b] = gt_labels_s[b][keep_idx_s]
                gt_scores_s[b] = gt_scores_s[b][keep_idx_s]
            
            if self.ssl_with_co_rectify:
                for b in range(len(gt_labels_s_2)):
                    assert gt_labels_s_2[b].shape[0] == gt_scores_s_2[b].shape[0]
                    keep_idx_s_2 = gt_scores_s_2[b] >= self.ssl_gt_box_low_thr
                    gt_bboxes_s_2[b] = gt_bboxes_s_2[b][keep_idx_s_2]
                    gt_labels_s_2[b] = gt_labels_s_2[b][keep_idx_s_2]
                    gt_scores_s_2[b] = gt_scores_s_2[b][keep_idx_s_2]

        if self.ssl_gt_box_low_thr > 0:
            num_pos_post = sum([(s >= self.ssl_gt_box_low_thr).sum().item() for s in gt_scores_s])
            num_pos_post2 = sum([(s >= self.ssl_gt_box_low_thr).sum().item() for s in gt_scores_s_2])
        else:
            num_pos_post = sum([len(s) for s in gt_scores_s])
            num_pos_post2 = sum([len(s) for s in gt_scores_s_2])

        losses = dict()
        losses.update({'num_pos_pre': img_s.new_tensor(num_pos_pre)})
        losses.update({'num_pos_post': img_s.new_tensor(num_pos_post)})
        if self.ssl_with_co_rectify:
            losses.update({'num_pos_pre2': img_s_2.new_tensor(num_pos_pre2)})
            losses.update({'num_pos_post2': img_s_2.new_tensor(num_pos_post2)})

        ## convert soft gt_labels to hard gt_labels
        for i, l in enumerate(gt_labels_s):
            if len(l) > 0 and l.dim() > 1:
                assert l.size(1) == self.roi_head.bbox_head.num_classes + 1
                gt_labels_s[i] = l[:, :-1].max(dim=1)[1]
        gt_bboxes_s = [g.clamp_(min=0, max=1e9) for g in gt_bboxes_s]
        if self.ssl_with_co_rectify:
            for i, l in enumerate(gt_labels_s_2):
                if len(l) > 0 and l.dim() > 1:
                    assert l.size(1) == self.roi_head.bbox_head.num_classes + 1
                    gt_labels_s_2[i] = l[:, :-1].max(dim=1)[1]
            gt_bboxes_s_2 = [g.clamp_(min=0, max=1e9) for g in gt_bboxes_s_2]
        
        num_mixup = 0
        num_mosaic = 0
        num_mixup_2 = 0
        num_mosaic_2 = 0

        if self.ssl_with_mixup or self.ssl_with_mosaic or self.ssl_with_mixup_labeled or self.ssl_with_mosaic_labeled:
            assert self.train_cfg['rcnn']['assigner']['use_soft_label'] == True

            candidate_aug_type_unlabel = []
            if self.ssl_with_mixup:
                candidate_aug_type_unlabel.append('ssl_with_mixup')
            if self.ssl_with_mosaic:
                candidate_aug_type_unlabel.append('ssl_with_mosaic')
            
            candidate_aug_type_label = []
            if self.ssl_with_mixup_labeled:
                candidate_aug_type_label.append('ssl_with_mixup')
            if self.ssl_with_mosaic_labeled:
                candidate_aug_type_label.append('ssl_with_mosaic')
            
            gt_labels_s = [g.new_zeros(g.size(0), num_classes + 1, dtype=torch.float32).scatter_(1, g.view(-1, 1), 1)
                            for g in gt_labels_s]
            if self.ssl_with_co_rectify:
                gt_labels_s_2 = [g.new_zeros(g.size(0), num_classes + 1, dtype=torch.float32).scatter_(1, g.view(-1, 1), 1)
                                for g in gt_labels_s_2]

            ## mixup between unlabel_image and label_image
            if len(candidate_aug_type_unlabel) > 0 and len(unlabeled_img_ids) > 0 and len(labeled_img_ids) > 0:
                for i in unlabeled_img_ids:
                    if np.random.rand() > 0.5:
                        continue
                    j = np.random.choice(labeled_img_ids)
                    aug_type = np.random.choice(candidate_aug_type_unlabel)
                    num_mixup += int(aug_type == 'ssl_with_mixup')
                    num_mosaic += int(aug_type == 'ssl_with_mosaic')
                    img_s[[i], ...], gt_bboxes_s[i], gt_labels_s[i] = self._mixup_mosaic_aug(
                                img_s, gt_bboxes_s, gt_labels_s, i, j, aug_type, alpha=1.0)
                
                if self.ssl_with_co_rectify:
                    for i in unlabeled_img_ids:
                        if np.random.rand() > 0.5:
                            continue
                        j = np.random.choice(labeled_img_ids)
                        aug_type = np.random.choice(candidate_aug_type_unlabel)
                        num_mixup_2 += int(aug_type == 'ssl_with_mixup')
                        num_mosaic_2 += int(aug_type == 'ssl_with_mosaic')
                        img_s_2[[i], ...], gt_bboxes_s_2[i], gt_labels_s_2[i] = self._mixup_mosaic_aug(
                                    img_s_2, gt_bboxes_s_2, gt_labels_s_2, i, j, aug_type, alpha=1.0)
                

            ## mixup between label_images
            if len(candidate_aug_type_label) > 0 and len(labeled_img_ids) > 0:
                for i in labeled_img_ids:
                    if np.random.rand() > 0.5:
                        continue
                    j = np.random.choice(list(set(labeled_img_ids) - {i}))
                    aug_type = np.random.choice(candidate_aug_type_label)
                    num_mixup += int(aug_type == 'ssl_with_mixup')
                    num_mosaic += int(aug_type == 'ssl_with_mosaic')
                    img_s[[i], ...], gt_bboxes_s[i], gt_labels_s[i] = self._mixup_mosaic_aug(
                                img_s, gt_bboxes_s, gt_labels_s, i, j, aug_type, alpha=1.0)
                
                if self.ssl_with_co_rectify:
                    for i in labeled_img_ids:
                        if np.random.rand() > 0.5:
                            continue
                        j = np.random.choice(list(set(labeled_img_ids) - {i}))
                        aug_type = np.random.choice(candidate_aug_type_label)
                        num_mixup_2 += int(aug_type == 'ssl_with_mixup')
                        num_mosaic_2 += int(aug_type == 'ssl_with_mosaic')
                        img_s_2[[i], ...], gt_bboxes_s_2[i], gt_labels_s_2[i] = self._mixup_mosaic_aug(
                                    img_s_2, gt_bboxes_s_2, gt_labels_s_2, i, j, aug_type, alpha=1.0)
        else:
            self.train_cfg['rcnn']['assigner'].get('use_soft_label', False) == False

        losses.update({'num_mixup': img_s.new_tensor(num_mixup)})
        losses.update({'num_mosaic': img_s.new_tensor(num_mosaic)})
        if self.ssl_with_co_rectify:
            losses.update({'num_mixup_2': img_s.new_tensor(num_mixup_2)})
            losses.update({'num_mosaic_2': img_s.new_tensor(num_mosaic_2)})

        x_s = self.extract_feat(img_s, idx=1)

        if self.ssl_lambda_u == 1.0:
            if self.with_rpn:
                proposal_cfg = self.train_cfg.get('rpn_proposal',
                                                self.test_cfg.rpn)
                rpn_losses_s, proposal_list_s = self.rpn_head.forward_train(
                    x_s,
                    img_metas_s,
                    gt_bboxes_s,
                    gt_labels=None,
                    gt_bboxes_ignore=gt_bboxes_ignore_s,
                    proposal_cfg=proposal_cfg)                                  # [(2000, 5), ...] in batch length
                for key in rpn_losses_s:
                    losses.update({key+'_s': rpn_losses_s[key]})
                
            
            roi_losses_s = self.roi_head.forward_train(x_s, img_metas_s, proposal_list_s,
                                                    gt_bboxes_s, gt_labels_s,
                                                    gt_bboxes_ignore_s, None,
                                                    **kwargs)
            for key in roi_losses_s:
                losses.update({key+'_s': roi_losses_s[key]})
        else:
            # compute losses for labeled and unlabeled images respectively
            if self.with_rpn:
                proposal_cfg = self.train_cfg.get('rpn_proposal',
                                                self.test_cfg.rpn)
                rpn_losses_s_l, proposal_list_s_l = self.rpn_head.forward_train(
                    [x[labeled_img_ids, ...] for x in x_s],
                    [img_metas_s[i] for i in labeled_img_ids],
                    [gt_bboxes_s[i] for i in labeled_img_ids],
                    gt_labels=None,
                    gt_bboxes_ignore=[gt_bboxes_ignore_s[i] for i in labeled_img_ids] if gt_bboxes_ignore_s is not None else None,
                    proposal_cfg=proposal_cfg)   
                for key in rpn_losses_s_l:
                    for k in range(len(rpn_losses_s_l[key])):
                        rpn_losses_s_l[key][k] *= 0.5
                    losses.update({key+'_s_l': rpn_losses_s_l[key]})

                rpn_losses_s_u, proposal_list_s_u = self.rpn_head.forward_train(
                    [x[unlabeled_img_ids, ...] for x in x_s],
                    [img_metas_s[i] for i in unlabeled_img_ids],
                    [gt_bboxes_s[i] for i in unlabeled_img_ids],
                    gt_labels=None,
                    gt_bboxes_ignore=[gt_bboxes_ignore_s[i] for i in unlabeled_img_ids] if gt_bboxes_ignore_s is not None else None,
                    proposal_cfg=proposal_cfg)  
                for key in rpn_losses_s_u:
                    for k in range(len(rpn_losses_s_u[key])):
                        rpn_losses_s_u[key][k] *= self.ssl_lambda_u * 0.5
                    losses.update({key+'_s_u': rpn_losses_s_u[key]})
                
            
            roi_losses_s_l = self.roi_head.forward_train(
                            [x[labeled_img_ids, ...] for x in x_s],
                            [img_metas_s[i] for i in labeled_img_ids],
                            proposal_list_s_l,
                            [gt_bboxes_s[i] for i in labeled_img_ids],
                            [gt_labels_s[i] for i in labeled_img_ids],
                            [gt_bboxes_ignore_s[i] for i in labeled_img_ids] if gt_bboxes_ignore_s is not None else None, 
                            None,
                            **kwargs)
            for key in roi_losses_s_l:
                if isinstance(roi_losses_s_l[key], (tuple, list)):
                    for k in range(len(roi_losses_s_l[key])):
                        roi_losses_s_l[key][k] *= 0.5
                else:
                    roi_losses_s_l[key] *= 0.5
                losses.update({key+'_s_l': roi_losses_s_l[key]})

            roi_losses_s_u = self.roi_head.forward_train(
                            [x[unlabeled_img_ids, ...] for x in x_s],
                            [img_metas_s[i] for i in unlabeled_img_ids],
                            proposal_list_s_u,
                            [gt_bboxes_s[i] for i in unlabeled_img_ids],
                            [gt_labels_s[i] for i in unlabeled_img_ids],
                            [gt_bboxes_ignore_s[i] for i in unlabeled_img_ids] if gt_bboxes_ignore_s is not None else None, 
                            None,
                            **kwargs)
            for key in roi_losses_s_u:
                if isinstance(roi_losses_s_u[key], (tuple, list)):
                    for k in range(len(roi_losses_s_u[key])):
                        roi_losses_s_u[key][k] *= self.ssl_lambda_u * 0.5
                else:
                    roi_losses_s_u[key] *= self.ssl_lambda_u * 0.5
                losses.update({key+'_s_u': roi_losses_s_u[key]})
            
        
        if self.ssl_with_co_rectify:
            assert self.ssl_lambda_u == 1.0, 'not implemented for other lambda_u now'

            x_s_2 = self.extract_feat(img_s_2, idx=2)
            if self.with_rpn:
                rpn_losses_s_2, proposal_list_s_2 = self.rpn_head2.forward_train(
                    x_s_2,
                    img_metas_s_2,
                    gt_bboxes_s_2,
                    gt_labels=None,
                    gt_bboxes_ignore=gt_bboxes_ignore_s_2,
                    proposal_cfg=proposal_cfg)                                  # [(2000, 5), ...] in batch length
                for key in rpn_losses_s_2:
                    losses.update({key+'_s2': rpn_losses_s_2[key]})

            roi_losses_s_2 = self.roi_head2.forward_train(x_s_2, img_metas_s_2, proposal_list_s_2,
                                                    gt_bboxes_s_2, gt_labels_s_2,
                                                    gt_bboxes_ignore_s_2, None,
                                                    **kwargs)
            for key in roi_losses_s_2:
                losses.update({key+'_s2': roi_losses_s_2[key]})

        return losses
    
    def _mixup_mosaic_aug(self, img_s, gt_bboxes_s, gt_labels_s, i_unlabel, j_label, aug_type, alpha=1.0):
        if aug_type == 'ssl_with_mixup':
            lamb = np.random.beta(alpha, alpha)
            _img = lamb * img_s[[i_unlabel], ...] + (1 - lamb) * img_s[[j_label], ...]
            _gt_labels = torch.cat((gt_labels_s[i_unlabel] * lamb, gt_labels_s[j_label] * (1 - lamb)))
            _gt_bboxes = torch.cat((gt_bboxes_s[i_unlabel], gt_bboxes_s[j_label]))

        elif aug_type == 'ssl_with_mosaic':
            _, _, _h, _w = img_s.shape
            _img = img_s[[i_unlabel], ...]
            if np.random.randint(0, 2) == 1:    ## split top-down
                cy = np.random.randint(_h // 4, _h // 4 * 3)
                _img[0, :, cy:, :] = img_s[[j_label], :, cy:, :]
                gt_bboxes_i, gt_labels_i = self._clip_gt(gt_bboxes_s[i_unlabel], gt_labels_s[i_unlabel], 0, 0, _w, cy)
                gt_bboxes_j, gt_labels_j = self._clip_gt(gt_bboxes_s[j_label], gt_labels_s[j_label], 0, cy, _w, _h)
                _gt_bboxes = torch.cat((gt_bboxes_i, gt_bboxes_j))
                _gt_labels = torch.cat((gt_labels_i, gt_labels_j))
            else:                               ## split left-right
                #cx = int(gt_bboxes[i][np.random.choice(list(range(len(gt_labels[i]))))][2])
                cx = np.random.randint(_w // 4, _w // 4 * 3)
                _img[0, :, :, cx:] = img_s[[j_label], :, :, cx:]
                gt_bboxes_i, gt_labels_i = self._clip_gt(gt_bboxes_s[i_unlabel], gt_labels_s[i_unlabel], 0, 0, cx, _h)
                gt_bboxes_j, gt_labels_j = self._clip_gt(gt_bboxes_s[j_label], gt_labels_s[j_label], cx, 0, _w, _h)
                _gt_bboxes = torch.cat((gt_bboxes_i, gt_bboxes_j))
                _gt_labels = torch.cat((gt_labels_i, gt_labels_j))
        return _img, _gt_bboxes, _gt_labels

    def _clip_gt(self, _gt_bboxes, _gt_labels, x1_lim, y1_lim, x2_lim, y2_lim):
        gt_bboxes = _gt_bboxes.clone()
        gt_labels = _gt_labels.clone()
        wh = gt_bboxes[:, [2, 3]] - gt_bboxes[:, [0, 1]]
        gt_bboxes[:, [0, 2]] = gt_bboxes[:, [0, 2]].clamp(x1_lim, x2_lim)
        gt_bboxes[:, [1, 3]] = gt_bboxes[:, [1, 3]].clamp(y1_lim, y2_lim)

        #keep = (gt_bboxes[:, 2] > gt_bboxes[:, 0]) & (gt_bboxes[:, 3] > gt_bboxes[:, 1])
        wh_2 = gt_bboxes[:, [2, 3]] - gt_bboxes[:, [0, 1]]
        keep = (wh_2[:, 0] > wh[:, 0] / 4.) & (wh_2[:, 1] > wh[:, 1] / 4.)
        return gt_bboxes[keep], gt_labels[keep]

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, proposals_additonal=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)             # list[(1000, 5)]
        else:
            proposal_list = proposals
        
        if proposals_additonal is not None:
            if isinstance(proposals_additonal, list):
                proposals_additonal = proposals_additonal[0]
            proposals_additonal = torch.tensor(proposals_additonal, device=img.device, dtype=img.dtype)

            assert len(img_metas) == 1 and img_metas[0]['flip'] == False
            proposals_additonal[:, :4] = proposals_additonal[:, :4] * proposals_additonal.new_tensor(img_metas[0]['scale_factor'])

            assert proposals_additonal.dim() == 2 and proposals_additonal.size(1) == 5
            proposal_list = [torch.cat((proposal_list[0], proposals_additonal))]
            
        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head1.aug_test_rpn(x, img_metas)
        return self.roi_head1.aug_test(
            x, proposal_list, img_metas, rescale=rescale)
