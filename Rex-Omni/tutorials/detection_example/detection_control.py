#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Rex Omni 逐帧检测验证脚本（无追踪）
单独验证检测模型效果，不进行时序关联
"""

import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import re
import sys
from collections import defaultdict
import cv2

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from rex_omni import RexOmniWrapper

from sahi.slicing import slice_image

class DetectionVisualizer:
    """简单的检测可视化器（无需追踪）"""
    
    def __init__(self, font_path: str = None):
        self.font_path = font_path
        # 为不同类别预定义颜色（可扩展）
        self.class_colors = {
            1: (255, 0, 0),    # car - 红色
            2: (0, 255, 0),    # plane - 绿色
            3: (0, 0, 255),    # ship - 蓝色
            4: (255, 255, 0),  # train - 黄色
        }
    
    def draw_detections(
        self,
        image: Image.Image,
        detections: list,
        font_size: int = 20,
        line_width: int = 3,
    ) -> Image.Image:
        """
        绘制检测结果
        
        Args:
            image: 输入图片
            detections: 检测列表，格式 [{'bbox': [x1,y1,x2,y2], 'score': 0.9, 'class_id': 1, 'class_name': 'car'}, ...]
        """
        vis_image = image.copy()
        draw = ImageDraw.Draw(vis_image)
        
        # 加载字体
        try:
            if self.font_path and os.path.exists(self.font_path):
                font = ImageFont.truetype(self.font_path, font_size)
            else:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            score = det['score']
            class_id = det['class_id']
            class_name = det['class_name']
            
            # 获取颜色（如果没有预定义则使用默认红色）
            color = self.class_colors.get(class_id, (255, 0, 0))
            
            # 绘制边界框
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
            
            # 绘制标签（类别:置信度）
            label = f"{class_name}:{score:.2f}"
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            
            # 标签背景框
            bg_coords = [x1, y1 - text_h - 6, x1 + text_w + 8, y1]
            if bg_coords[1] < 0:  # 防止越界
                bg_coords = [x1, y1, x1 + text_w + 8, y1 + text_h + 6]
            
            draw.rectangle(bg_coords, fill=color, outline=color)
            draw.text((bg_coords[0] + 4, bg_coords[1] + 3), label, 
                     fill=(255, 255, 255), font=font)
        
        return vis_image


def main():
    # ==================== 配置参数 ====================
    # ✅ 修改：指定您的视频序列路径
    image_folder = "/home/REXOMNI-125-main/Rex-Omni/dataset/visdrone2019/264/sahi_for_test"
    
    # 检测结果保存目录
    SAVE_RESULTS_DIR = "/home/REXOMNI-125-main/Rex-Omni/tracking_results"
    os.makedirs(SAVE_RESULTS_DIR, exist_ok=True)
    
    # 可视化输出目录
    VIS_OUTPUT_DIR = "/home/REXOMNI-125-main/Rex-Omni/output/sequence_sahi"
    os.makedirs(VIS_OUTPUT_DIR, exist_ok=True)
    
    # 自动提取序列名称
    sequence_name = os.path.basename(os.path.normpath(image_folder))
    
    # 类别配置
    # categories = ["car", "plane", "ship", "train"]  # 多类别示例
    categories = ["car"]  # 当前单类别
    CLASS_ID_MAP = {"car": 1}
    
    # 初始化结果存储列表
    detection_results = []
    
    # ==================== 加载模型 ====================
    print("加载 Rex Omni 模型...")
    model_path = "/home/weights"
    rex_model = RexOmniWrapper(
        model_path=model_path,
        backend="transformers",
        max_tokens=4096,
        temperature=0.0,
        top_p=0.05,
        top_k=1,
        repetition_penalty=1.05,
    )
    
    # ==================== 初始化可视化器 ====================
    visualizer = DetectionVisualizer()
    
    # ==================== 读取图片 ====================
    image_files = sorted(
        [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".png"))],
        key=lambda x: int(re.search(r"\d+", x).group()) if re.search(r"\d+", x) else x,
    )
    
    if not image_files:
        raise ValueError(f"在 {image_folder} 中未找到任何图片文件！")
        
    print(f"共找到 {len(image_files)} 帧图片")
    print(f"序列名称: {sequence_name}")
    print(f"类别配置: {categories}")
    
    # ==================== 逐帧处理 ==================== 
    for frame_idx, image_file in enumerate(image_files, 1):
        print(f"\r处理帧: {frame_idx}/{len(image_files)}", end="", flush=True)
        
        # 加载图片
        image_path = os.path.join(image_folder, image_file)
        image = Image.open(image_path).convert("RGB")

        # resized_image = image.resize(
        #     (1024, 1024), 
        #     Image.Resampling.LANCZOS  # 对于PIL 9.0+
        #     # Image.LANCZOS  # 对于旧版本PIL
        # )
    
        # # 保存结果
        # resized_image.save(image_path, quality=95)  # quality参数对JPEG有效
        
        #SAHI登场！
        output_file_name = "slice"
        output_dir = "/home/REXOMNI-125-main/Rex-Omni/dataset/sahi_sliced"
        slice_image_result = slice_image(
            image=image_path,
            output_file_name=output_file_name,
            output_dir=output_dir,
            slice_height=256,
            slice_width=256,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
        )

        # 模型推理
        results = rex_model.inference(images=image, task="detection", categories=categories)
        
        # 准备检测结果
        detections = []
        detection_list = []  # 用于可视化和保存
        
        if results[0]["success"]:
            predictions = results[0]["extracted_predictions"]
            confidence = results[0]["token_confidences"]
            
            # 提取检测框和置信度
            i = 4
            for cat, items in predictions.items():
                class_id = CLASS_ID_MAP.get(cat, 1)
                for item in items:
                    if item["type"] == "box":
                        bbox = item['coords']  # [x1, y1, x2, y2]
                        # 计算置信度（取4个坐标的平均值）
                        score = sum(confidence[i:i+4])/4 if i+4 < len(confidence) else 0.5
                        i += 5
                        
                        # 过滤低分框（阈值可根据需求调整）
                        if score > 0.3:
                            detections.append([*bbox, score])
                            detection_list.append({
                                'bbox': bbox,
                                'score': score,
                                'class_id': class_id,
                                'class_name': cat
                            })
        
        # ==================== 保存检测结果 ====================
        for det in detection_list:
            x1, y1, x2, y2 = det['bbox']
            w, h = x2 - x1, y2 - y1
            detection_results.append([
                frame_idx,          # frame_id
                -1,                 # track_id (检测模式用-1表示)
                x1,                 # x
                y1,                 # y
                w,                  # w
                h,                  # h
                det['score'],       # confidence
                det['class_id'],    # class_id
                1                   # visibility
            ])
        
        # ==================== 可视化 ====================
        vis_image = visualizer.draw_detections(
            image=image,
            detections=detection_list,
            font_size=20,
            line_width=3
        )
        
        # 保存可视化结果
        vis_image.save(os.path.join(VIS_OUTPUT_DIR, f"frame_{frame_idx:03d}.jpg"))
    
    # ==================== 保存结果文件 ====================
    result_file_path = os.path.join(SAVE_RESULTS_DIR, f"{sequence_name}_detections.txt")
    with open(result_file_path, 'w+') as fid:
        for result in detection_results:
            fid.write('%d,%d,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%d,%d\n' % tuple(result))
    
    print(f"\n✅ 检测完成！")
    print(f"   - 结果文件: {result_file_path}")
    print(f"   - 可视化帧: {VIS_OUTPUT_DIR}/frame_*.jpg")
    print(f"   - 总检测框数: {len(detection_results)} 个")
    print(f"   - 处理帧数: {len(image_files)} 帧")


if __name__ == "__main__":
    main()