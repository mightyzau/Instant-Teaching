#coding=utf-8
# E.g., split coco2017_train to 5% labeled images and 95% unlabeled images.

import json
import random
import os

def get_imgId_to_anns(data):
    imgId_to_anns = {}
    for ann in data['annotations']:
        image_id = ann['image_id']
        if image_id not in imgId_to_anns:
            imgId_to_anns[image_id] = []
        imgId_to_anns[image_id].append(ann)
    return imgId_to_anns

def get_imgId_to_imgInfo(data):
    imgId_to_imgInfo = {}
    for img_info in data['images']:
        image_id = img_info['id']
        imgId_to_imgInfo[image_id] = img_info
    return imgId_to_imgInfo


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('coco_ann', type=str)
    parser.add_argument('percent', type=int)
    parser.add_argument('--fold_num', type=int)
    args = parser.parse_args()
    
    data = json.load(open(args.coco_ann))

    imgId_to_anns = get_imgId_to_anns(data)

    img_infos = data['images']
    random.shuffle(img_infos)

    num_labeled = len(img_infos) * args.percent // 100

    if args.fold_num is not None:
        labeled_file_name = os.path.splitext(args.coco_ann)[0] + '_{}_labeled_fold-{}.json'.format(args.percent, args.fold_num)
        unlabeled_file_name = os.path.splitext(args.coco_ann)[0] + '_{}_unlabeled_fold-{}.json'.format(100 - args.percent, args.fold_num)
    else:
        labeled_file_name = os.path.splitext(args.coco_ann)[0] + '_{}_labeled.json'.format(args.percent)
        unlabeled_file_name = os.path.splitext(args.coco_ann)[0] + '_{}_unlabeled.json'.format(100 - args.percent)
    

    labeled_data = {'images': [], 'annotations': [], 'categories': data['categories']}
    for img_info in img_infos[:num_labeled]:
        labeled_data['images'].append(img_info)
        img_id = img_info['id']
        if img_id in imgId_to_anns:
            anns = imgId_to_anns[img_id]
            labeled_data['annotations'].extend(anns)
    with open(labeled_file_name, 'w') as f:
        json.dump(labeled_data, f)
    print('write: {}, {} images'.format(labeled_file_name, len(labeled_data['images'])))

    
    unlabeled_data = {'images': [], 'annotations': [], 'categories': data['categories']}
    ann_id = 1
    for img_info in img_infos[num_labeled:]:
        img_info['is_unlabeled'] = True
        # add a dummy ann box
        img_id = img_info['id'] 
        x1, y1, w, h = 0, 0, 200, 200
        unlabeled_data['annotations'].append({
            'bbox': [x1, y1, w, h],
            'segmentation': [[x1, y1, x1+w, y1, x1+w, y1+h, x1, y1+h, x1, y1]],
            'image_id': img_id,
            'id': ann_id,
            'score': 0.01,   # low value, to be filtered in semi supervised training
            'area': 200 * 200,
            'category_id': 1,
        })
        ann_id += 1

        unlabeled_data['images'].append(img_info)
    with open(unlabeled_file_name, 'w') as f:
        json.dump(unlabeled_data, f)
    print('write: {}, {} images'.format(unlabeled_file_name, len(unlabeled_data['images'])))  
