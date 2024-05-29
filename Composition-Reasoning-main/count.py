# input: bbox, mask, category from SAM, depth from DAM
# output: instance-level information, including category, coordiante(center of bbox, depth)
import json
# import ipdb
import cv2
import torch
import os
def count(sam_file_path, sam_d_file_path):
    ins_infos = []
    # sam_info = torch.load(sam_file_path)
    with open(sam_d_file_path, 'r') as file:
        sam_d_info = json.load(file)
    for idx, item in enumerate(sam_d_info):
        if idx == 0: # background
            continue 
        info = {}
        info['category'] = item['label']
        info['bbox'] = item['box']
        x1, y1, x2, y2 = info['bbox']
        info['position'] = ((x1 + x2) / 2, (y1 + y2) / 2)
        ins_infos.append(info)
    print('There are', idx, info['category']+ 's')    
    return ins_infos, idx
if __name__ == '__main__':
    # image_path = '/mnt/hdd1/Grounded-Segment-Anything/assets/test.jpg'
    sam_file_path = '/home/mhuo/Grounded-Segment-Anything/outputs/masks.pt'
    sam_d_file_path = '/home/mhuo/Grounded-Segment-Anything/outputs/mask.json'
    ins_info,count = count(sam_d_file_path,sam_d_file_path)
    print(count)