#coding=utf-8
import itertools
import logging
import os.path as osp
import tempfile
import torch
import copy
import random

import mmcv
import numpy as np
from mmcv.utils import print_log
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from terminaltables import AsciiTable
from torch.utils.data import Dataset

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.pipelines import Compose


@DATASETS.register_module()
class SSLCocoDataset(Dataset):
    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
    
    def __init__(self, 
                 ann_file_l, 
                 ann_file_u,
                 pipeline_l, 
                 pipeline_u,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 **kwargs
        ):
        self.ann_file_l = ann_file_l
        self.ann_file_u = ann_file_u
        self.pipeline_l = pipeline_l
        self.pipeline_u = pipeline_u

        self.data_root = data_root
        if isinstance(img_prefix, (list, tuple)):
            self.img_prefix_l, self.img_prefix_u = img_prefix
        else:
            self.img_prefix_l, self.img_prefix_u = img_prefix, img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.CLASSES = self.get_classes(classes)

        # join paths if data_root is specified
        assert self.data_root is None

        # load annotations (and proposals)
        self.coco_l = COCO(self.ann_file_l)
        self.coco_u = COCO(self.ann_file_u)
        self.data_infos_l = self.load_annotations(self.coco_l)
        self.data_infos_u = self.load_annotations(self.coco_u)
        assert len(self.data_infos_u) >= len(self.data_infos_l)
        self.data_ratio = len(self.data_infos_u) // len(self.data_infos_l)

        self.cat_ids = self.coco_l.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

        # filter data infos if classes are customized
        assert not self.custom_classes
        assert self.proposal_file is None
        self.proposals = None

        # filter images too small
        if not test_mode:
            valid_inds_l = self._filter_imgs(self.coco_l, self.data_infos_l)
            self.data_infos_l = [self.data_infos_l[i] for i in valid_inds_l]
            valid_inds_u = self._filter_imgs(self.coco_u, self.data_infos_u)
            self.data_infos_u = [self.data_infos_u[i] for i in valid_inds_u]

            print('\n')
            print('\tdata_infos_l: {} images'.format(len(self.data_infos_l)))
            print('\tdata_infos_u: {} images'.format(len(self.data_infos_u)))

        # set group flag for the sampler
        if not self.test_mode:
            #self.flag_l = self._set_group_flag(self.data_infos_l)
            #self.flag_u = self._set_group_flag(self.data_infos_u)
            self.flag = np.zeros(len(self.data_infos_l), dtype=np.uint8)

        # processing pipeline
        ## pipeline_l: list of original pipeline in format: [weak_aug, strong_aug_1, strong_aug_2, ...]
        assert isinstance(self.pipeline_l, (list, tuple))
        assert isinstance(self.pipeline_u, (list, tuple))
        assert len(self.pipeline_l) == len(self.pipeline_u)
        assert len(self.pipeline_l) in [2, 3]
        self.pipeline_weak_aug_l = Compose(self.pipeline_l[0][:-1])
        self.pipeline_weak_aug_u = Compose(self.pipeline_u[0][:-1])
        self.pipeline_strong_aug_l = Compose(self.pipeline_l[1][:-1])
        self.pipeline_strong_aug_u = Compose(self.pipeline_u[1][:-1])
        if len(self.pipeline_l) == 3:
            self.pipeline_strong_aug_2_l = Compose(self.pipeline_l[2][:-1])
            self.pipeline_strong_aug_2_u = Compose(self.pipeline_u[2][:-1])
        assert self.pipeline_l[-1][-1]['type'] == self.pipeline_u[-1][-1]['type'] == 'CollectList'
        self.pipeline_collect = Compose(pipeline_l[-1][-1:])
  
    def _set_group_flag(self, data_infos):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        flag = np.zeros(len(data_infos), dtype=np.uint8)
        for i in range(len(data_infos)):
            img_info = data_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                flag[i] = 1
        return flag
    
    @classmethod
    def get_classes(cls, classes=None):
        """Get class names of current dataset
        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
        """
        if classes is None:
            cls.custom_classes = False
            return cls.CLASSES

        cls.custom_classes = True
        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        return class_names

    def pre_pipeline(self, results, is_labeled=True):
        """Prepare results dict for pipeline"""
        if is_labeled:
            results['img_prefix'] = self.img_prefix_l
        else:
            results['img_prefix'] = self.img_prefix_u
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
    
    def _rand_another(self, flag, idx):
        """Get another random index from the same group as the given index"""
        pool = np.where(flag == flag[idx])[0]
        return np.random.choice(pool)
    
    def get_cat_ids(self, idx):
        """Get COCO category ids by index.
        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        img_id_l = self.data_infos_l[idx]['id']
        ann_ids_l = self.coco_l.get_ann_ids(img_ids=[img_id_l])
        ann_info_l = self.coco_l.load_anns(ann_ids_l)
        return [ann['category_id'] for ann in ann_info_l]
    
    def __len__(self):
        """Total number of samples of data"""
        return len(self.data_infos_l)

    def __getitem__(self, idx):
        """Get training/test data after pipeline

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                True).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """
        idx_l = idx
        idx_u = idx * self.data_ratio + np.random.randint(self.data_ratio)
        while True:
            ## extract labeled images
            img_info_l = self.data_infos_l[idx_l]
            ann_info_l = self.get_ann_info(self.data_infos_l, self.coco_l, idx_l)
            results_l_w_aug = dict(img_info=img_info_l, ann_info=ann_info_l)
            results_l_s_aug = copy.deepcopy(results_l_w_aug)
            if hasattr(self, 'pipeline_strong_aug_2_l'):
                results_l_s_aug_2 = copy.deepcopy(results_l_w_aug)
            
            self.pre_pipeline(results_l_w_aug, True)
            self.pre_pipeline(results_l_s_aug, True)
            if hasattr(self, 'pipeline_strong_aug_2_l'):
                self.pre_pipeline(results_l_s_aug_2, True)

            data_l_w = self.pipeline_weak_aug_l(results_l_w_aug)
            data_l_s = self.pipeline_strong_aug_l(results_l_s_aug)
            if hasattr(self, 'pipeline_strong_aug_2_l'):
                data_l_s_2 = self.pipeline_strong_aug_2_l(results_l_s_aug_2)

            if (data_l_w is None) or (data_l_s is None):
                idx_l = self._rand_another(self.flag_l, idx_l)     
                continue       

            ## extract from unlabeled images
            img_info_u = self.data_infos_u[idx_u]
            ann_info_u = self.get_ann_info(self.data_infos_u, self.coco_u, idx_u)
            results_u_w_aug = dict(img_info=img_info_u, ann_info=ann_info_u)
            results_u_s_aug = copy.deepcopy(results_u_w_aug)
            if hasattr(self, 'pipeline_strong_aug_2_u'):
                results_u_s_aug_2 = copy.deepcopy(results_u_w_aug)
            
            self.pre_pipeline(results_u_w_aug, False)
            self.pre_pipeline(results_u_s_aug, False)
            if hasattr(self, 'pipeline_strong_aug_2_u'):
                self.pre_pipeline(results_u_s_aug_2, False)

            data_u_w = self.pipeline_weak_aug_u(results_u_w_aug)
            data_u_s = self.pipeline_strong_aug_u(results_u_s_aug)
            if hasattr(self, 'pipeline_strong_aug_2_u'):
                data_u_s_2 = self.pipeline_strong_aug_2_u(results_u_s_aug_2)

            if (data_u_w is None) or (data_u_s is None):
                idx_u = self._rand_another(self.flag_u, idx_u)     
                continue  
            
            if hasattr(self, 'pipeline_strong_aug_2_u'):
                return self.pipeline_collect([data_l_w, data_l_s, data_l_s_2, data_u_w, data_u_s, data_u_s_2])
            
            return self.pipeline_collect([data_l_w, data_l_s, data_u_w, data_u_s])

    def load_annotations(self, coco_obj):
        img_ids = coco_obj.get_img_ids()
        data_infos = []
        for img_id in img_ids:
            info = coco_obj.load_imgs([img_id])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
        return data_infos

    def get_ann_info(self, data_infos, coco_obj, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        img_id = data_infos[idx]['id']
        ann_ids = coco_obj.get_ann_ids(img_ids=[img_id])
        ann_info = coco_obj.load_anns(ann_ids)
        return self._parse_ann_info(data_infos[idx], ann_info)

    def _filter_imgs(self, coco_obj, data_infos, min_size=32):
        """Filter images too small or without ground truths."""
        ids_with_ann = set(_['image_id'] for _ in coco_obj.anns.values())

        valid_inds = []
        for i, img_info in enumerate(data_infos):
            if self.filter_empty_gt and img_info['id'] not in ids_with_ann:
                #print('ignore empty groundtruth image')
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_scores = []                              # for pseudo boxes
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue

            x1, y1, w, h = ann['bbox']
            x2, y2 = x1 + w, y1 + h
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                score = ann.get('score', 1.0)       # labeled标注score默认为1.0
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_scores.append(score)
                if 'segmentation' in ann:
                    gt_masks_ann.append(ann['segmentation'])
                else:
                    gt_masks_ann.append([[x1, y1, x2, y1, x2, y2, x1, y2, x1, y1]])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            gt_scores = np.array(gt_scores, dtype=np.float32)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            gt_scores = np.array([], dtype=np.float32)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            scores=gt_scores,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann
