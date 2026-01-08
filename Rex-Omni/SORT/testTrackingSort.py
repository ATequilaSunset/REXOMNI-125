# [FINAL VERSION 3] - Fixes the DeprecationWarning for future NumPy compatibility.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from lib.utils.opts import opts
opt = opts().parse()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
import numpy as np
import time
import torch
from collections import defaultdict

from lib.models.stNet import get_det_net, load_model
from lib.dataset.coco_icpr import COCO
from lib.utils.decode import ctdet_decode
from lib.utils.post_process import generic_post_process
from lib.utils.sort import Sort
from progress.bar import Bar


def pre_process(image_tensor, scale=1):
    height, width = image_tensor.shape[2], image_tensor.shape[3]
    new_height = int(height * scale)
    new_width = int(width * scale)
    c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
    s = max(height, width) * 1.0
    meta = {'c': c, 's': s,
            'out_height': new_height,
            'out_width': new_width}
    return meta

def convert_dets_for_sort(dets_by_class, num_classes, conf_thres):
    dets_for_sort = {}
    for cls_id in range(1, num_classes + 1):
        list_index = cls_id - 1
        
        if list_index < len(dets_by_class):
            detections = dets_by_class[list_index]
        else:
            detections = []

        filtered_dets = []
        for item in detections:
            if 'score' in item and item['score'] > conf_thres:
                
                bbox_list = item['bbox'].tolist()
                # [THE FIX FOR THE WARNING] Extract the single element before converting to float.
                score_float = float(item['score'][0])
                filtered_dets.append(bbox_list + [score_float])
        
        if len(filtered_dets) > 0:
            dets_for_sort[cls_id] = np.array(filtered_dets, dtype=np.float32)
        else:
            dets_for_sort[cls_id] = np.empty((0, 5), dtype=np.float32)
            
    return dets_for_sort


def test(opt, split, modelPath):
    print(f"模型名称: {opt.model_name}")
    print(f"加载模型权重: {modelPath}")
    
    dataset = COCO(opt, split)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    num_classes = dataset.num_classes

    print("正在加载模型...")
    model = get_det_net({'hm': num_classes, 'wh': 2, 'reg': 2, 'dis': 2}, opt.model_name)
    model = load_model(model, modelPath)
    model = model.cuda()
    model.eval()
    print("模型加载完毕。")

    trackers = {} 

    saveTxt = opt.save_track_results
    if saveTxt:
        track_results_save_dir = os.path.join(opt.save_results_dir, 'trackingResults_SORT_25epoch_0.3_' + opt.model_name)
        if not os.path.exists(track_results_save_dir):
            os.makedirs(track_results_save_dir)
    
    file_folder_pre = 'INIT'
    fid = None
    bar = Bar('Processing', max=len(data_loader))
    
    total_processing_time = 0.0
    total_frames_processed = 0
    
    for ind, (img_id, pre_processed_images) in enumerate(data_loader):
        if 'file_name' not in pre_processed_images: continue

        torch.cuda.synchronize()
        start_time = time.time()

        file_name = pre_processed_images['file_name'][0]
        file_folder_cur = file_name.split('_')[0]
        
        if file_folder_cur != file_folder_pre:
            print(f"\n检测到新视频序列: {file_folder_cur}")
            if fid: fid.close()
            
            trackers = {} 
            for i in range(1, num_classes + 1):
                trackers[i] = Sort(max_age=opt.max_age, min_hits=3, iou_threshold=opt.iou_threshold if hasattr(opt, 'iou_threshold') else 0.3)

            im_count = 0
            if saveTxt:
                txt_path = os.path.join(track_results_save_dir, file_folder_cur + '.txt')
                fid = open(txt_path, 'w+')
            file_folder_pre = file_folder_cur

        image_tensor = pre_processed_images['input']
        
        with torch.no_grad():
            output = model(image_tensor.cuda(), training=False)[-1][1]
            hm, wh = output['hm'].sigmoid_(), output['wh']
            reg = output['reg'] if 'reg' in output else None
            dis = output['dis'] if 'dis' in output else None
            
            dets_feature_map = ctdet_decode(hm, wh, reg=reg, tracking=dis, num_classes=num_classes, K=opt.K)
            for k in dets_feature_map: dets_feature_map[k] = dets_feature_map[k].detach().cpu().numpy()

        meta = pre_process(image_tensor)
        dets_image_map_by_class = generic_post_process(
            dets_feature_map, [meta['c']], [meta['s']], 
            meta['out_height'] // opt.down_ratio, 
            meta['out_width'] // opt.down_ratio, 
            num_classes
        )
        
        dets_for_sort = convert_dets_for_sort(dets_image_map_by_class, num_classes, opt.conf_thres)
        
        im_count += 1
        
        for cls_id in range(1, num_classes + 1):
            if cls_id in trackers:
                tracked_objects = trackers[cls_id].update(dets_for_sort.get(cls_id, np.empty((0, 5))))
                
                if saveTxt and fid is not None and len(tracked_objects) > 0:
                    for d in tracked_objects:
                        x1, y1, x2, y2, track_id = d
                        w, h = x2 - x1, y2 - y1
                        line = f"{im_count},{int(track_id)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,{cls_id},1\n"
                        fid.write(line)
        
        torch.cuda.synchronize()
        end_time = time.time()
        total_processing_time += (end_time - start_time)
        total_frames_processed += 1
        
        bar.suffix = f"[{ind}/{len(data_loader)}]|ETA: {bar.eta_td}"
        bar.next()
    
    if fid: fid.close()
    bar.finish()

    if total_processing_time > 0:
        average_fps = total_frames_processed / total_processing_time
        print("\n\n" + "="*50)
        print("           算法性能测试结果 (SORT Version)")
        print("="*50)
        print(f"  总处理帧数: {total_frames_processed} 帧")
        print(f"  总耗时: {total_processing_time:.3f} 秒")
        print(f"  平均FPS (帧/秒): {average_fps:.2f}")
        print("="*50)
    else:
        print("\n未能处理任何帧，无法计算FPS。")

if __name__ == '__main__':
    if not hasattr(opt, 'conf_thres'): opt.conf_thres = 0.3
    if not hasattr(opt, 'max_age'): opt.max_age = 30
    if not hasattr(opt, 'split'): opt.split = 'val'
    
    if not os.path.exists(opt.save_results_dir):
        os.makedirs(opt.save_results_dir)

    modelPath = opt.load_model if opt.load_model != '' else './checkpoints/MP2Net.pth'
    
    test(opt, opt.split, modelPath)