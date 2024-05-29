# input: bbox, mask, category from SAM, depth from DAM
# output: instance-level information, including category, coordiante(center of bbox, depth)
import json
# import ipdb
import cv2
import torch
import os
def merge(sam_file_path, sam_d_file_path, dam_file_path):
    ins_infos = []
    sam_info = torch.load(sam_file_path)
    dam_info = torch.load(dam_file_path)
    depth_ins_wise = dam_info
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
        mask = sam_info[idx - 1].squeeze()
        print(mask.shape)
        info['mask'] = mask
        info['depth'] = dam_info[mask].mean().cpu().numpy().item()
        ins_infos.append(info)
        
        depth_ins_wise[mask] = info['depth']
    return ins_infos, depth_ins_wise
    
    
def vis_ins_depth(depth):
    
    cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
    
if __name__ == '__main__':
    image_path = '/mnt/hdd1/Grounded-Segment-Anything/assets/test.jpg'
    sam_file_path = '/home/mhuo/Grounded-Segment-Anything/outputs/masks.pt'
    sam_d_file_path = '/home/mhuo/Grounded-Segment-Anything/outputs/mask.json'
    dam_file_path = '/home/mhuo/Depth-Anything/output/depth.pt'
    ins_infos = merge(sam_file_path, sam_d_file_path, dam_file_path)
    print(ins_infos)
    
    output_file_path = os.path.join('/home/mhuo/Composition-Reasoning-main/', 'ins_infos1.pt')
    
    # Save ins_infos to a file
    torch.save(ins_infos, output_file_path)

    print(f"Saved ins_infos to {output_file_path}")