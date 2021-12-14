
This repo is the official implementation of ["Instant-Teaching: An End-to-End Semi-Supervised Object Detection Framework"](https://arxiv.org/abs/2103.11402).
```
@inproceedings{zhou2021instant,
  title={Instant-Teaching: An End-to-End Semi-Supervised Object Detection Framework},
  author={Zhou, Qiang and Yu, Chaohui and Wang, Zhibin and Qian, Qi and Li, Hao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4081--4090},
  year={2021}
}
```

The code is based on MMdetection Toolbox.


## 1. Install
```
conda create -n instant_teaching python=3.7 -y
conda activate instant_teaching
conda install pytorch=1.7.1 torchvision cudatoolkit=10.1 -c pytorch
pip install mmcv==1.2.4

git clone http://gitlab.alibaba-inc.com/jianchong.zq/InstantTeaching.git
cd InstantTeaching
pip install -v -e .
```


## 2. Prepare COCO Dataset for Semi-Supervised Learning
```
mkdir -p datasets/coco
ln -s [coco_2017_downloaded] datasets/coco/coco_2017

python projects/InstantTeaching/tools/split_coco.py datasets/coco/coco_2017/annotations/instances_train2017.json 10
```
Two files of _instances_train2017_10_labeled.json_ and _instances_train2017_90_unlabeled.json_ will be generated.


## 3. Training
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
    --nproc_per_node=8 --master_port=1234 tools/train.py \
    projects/InstantTeaching/configs/instant_teaching_two_stage_r50_coco_quick_10.py \
    --launcher pytorch \
    --work-dir ${output_dir} \
    --no-validate
```
