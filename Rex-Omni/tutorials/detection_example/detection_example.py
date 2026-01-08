#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Basic object detection example using Rex Omni
"""

import torch
from PIL import Image
from rex_omni import RexOmniVisualize, RexOmniWrapper

import numpy as np
import os
import re

import sys

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from SORT import Sort, TrackingVisualizer


def main():
    # ==================== 配置参数 ====================
    # ✅ 修改：指定您的视频序列路径
    image_folder = "/home/REXOMNI-125-main/Rex-Omni/Dataset/vis2019-mot-test-dev-mini/uav0000077_00720_v"

    # 自动解析序列信息
    img_basename = os.path.basename(os.path.normpath(image_folder)) 
    # 查找uav在字符串中的位置
    uav_index = img_basename.find('uav')
    if uav_index != -1 and len(img_basename) >= uav_index + 10:
        suffix = img_basename[uav_index + 7:uav_index + 10]  
    else:
        suffix = "001"  # 默认值
    sequence_name = os.path.basename(os.path.normpath(image_folder))

    # 获取Rex-Omni根目录（跳三级目录）
    rex_omni_root = os.path.dirname(os.path.dirname(os.path.dirname(image_folder))) # 得到 /home/REXOMNI-125-main/Rex-Omni
    output_root = os.path.join(rex_omni_root, "Output", "test")

    # 结果输出路径
    results_root = os.path.join(output_root, "results")
    SAVE_RESULTS_DIR = os.path.join(results_root, f"result{suffix}_1")
    os.makedirs(SAVE_RESULTS_DIR, exist_ok=True)
    # 可视化输出路径
    VIS_OUTPUT_DIR = os.path.join(output_root, f"vis{suffix}_1")
    os.makedirs(VIS_OUTPUT_DIR, exist_ok=True)
    # 调试切片输出路径 (Parallel to VIS_OUTPUT_DIR)
    # DEBUG_SLICES_DIR = os.path.join(output_root, f"debug_slices{suffix}")
    # os.makedirs(DEBUG_SLICES_DIR, exist_ok=True)



    # 类别配置
    # categories = ["car", "plane", "ship", "train"]  # 多类别示例
    categories = ["car"]  # 当前单类别
    CLASS_ID_MAP = {"car": 1}
    
    # 初始化结果存储列表
    tracking_results = []
    
    # ==================== 加载模型 ====================
    model_path = "/home/weights"
    rex_model = RexOmniWrapper(
        model_path=model_path,
        backend="transformers",
        max_tokens=4096,
        #temperature=0.0,
        #top_p=0.05,
        #top_k=1,
        repetition_penalty=1.05,
    )
    
    # ==================== SORT追踪器 ====================
    tracker = Sort(max_age=30, min_hits=5, iou_threshold=0.3)
    visualizer = TrackingVisualizer()
    
    # ==================== 读取图片 ====================
    image_files = sorted(
        [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".png"))],
        key=lambda x: int(re.search(r"\d+", x).group()) if re.search(r"\d+", x) else x,
    )
    
    if not image_files:
        raise ValueError(f"在 {image_folder} 中未找到任何图片文件！")
        
    print(f"共找到 {len(image_files)} 帧图片")
    print(f"序列名称: {sequence_name}")
    print(f"正在处理并保存结果到: {SAVE_RESULTS_DIR}/{sequence_name}.txt")
    
    # ==================== 逐帧处理 ==================== 
    for frame_idx, image_file in enumerate(image_files, 1):
        print(f"\r处理帧: {frame_idx}/{len(image_files)}", end="", flush=True)
        
        # 加载图片
        image_path = os.path.join(image_folder, image_file)
        image = Image.open(image_path).convert("RGB")
        
        # 模型推理
        results = rex_model.inference(images=image, task="detection", categories=categories)
        
        # 准备检测结果
        detections = []
        if results[0]["success"]:
            predictions = results[0]["extracted_predictions"]
            confidence = results[0]["token_confidences"]
            
            # 按testDis.py逻辑提取置信度
            i = 4
            for cat, items in predictions.items():
                class_id = CLASS_ID_MAP.get(cat, 1)
                for item in items:
                    if item["type"] == "box":
                        bbox = item['coords']  # [x1, y1, x2, y2]
                        # 计算平均置信度
                        score = sum(confidence[i:i+4])/4 if i+4 < len(confidence) else 0.5
                        i += 5
                        # 过滤低分框（可选，根据需求调整阈值）
                        if score > 0.3:
                            detections.append([*bbox, score])
        
        # 如果没有检测框，传入空数组
        if len(detections) == 0:
            detections = np.empty((0, 5))
        else:
            detections = np.array(detections)
        
        # 更新追踪器
        tracked_objects = tracker.update(detections)
        
        # ==================== 保存MOT格式结果 ====================
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = obj
            w, h = x2 - x1, y2 - y1
            # 为每个检测到的类别添加记录
            for cat in categories:
                class_id = CLASS_ID_MAP[cat]
                tracking_results.append([
                    frame_idx,      # frame_id
                    int(track_id),  # track_id
                    x1,             # x
                    y1,             # y
                    w,              # w
                    h,              # h
                    1,              # conf #?换成手算的置信度会不会更好些
                    class_id,       # class_id
                    1               # visibility
                ])
        
        # ==================== 可视化 ====================
        vis_image = visualizer.draw_tracks(
            image=image,
            tracks=tracked_objects,
            font_size=20,
            line_width=3,
            show_trajectory=True
        )
        
        # 保存可视化结果
        vis_image.save(os.path.join(VIS_OUTPUT_DIR, f"frame_{frame_idx:03d}.jpg"))
    
    # ==================== 保存评估文件 ====================
    result_file_path = os.path.join(SAVE_RESULTS_DIR, f"{sequence_name}.txt")
    with open(result_file_path, 'w+') as fid:
        for result in tracking_results:
            fid.write('%d,%d,%0.2f,%0.2f,%0.2f,%0.2f,%d,%d,%d\n' % tuple(result))
    
    print(f"\n✅ 处理完成！")
    print(f"   - 结果文件: {result_file_path}")
    print(f"   - 可视化帧: {VIS_OUTPUT_DIR}/frame_*.jpg")
    print(f"   - 总记录数: {len(tracking_results)} 条")


if __name__ == "__main__":
    main()