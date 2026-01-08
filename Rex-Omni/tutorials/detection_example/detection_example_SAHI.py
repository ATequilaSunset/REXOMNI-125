#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
项目名称: Rex-Omni 多目标跟踪系统 (Rex-Omni MOT System)
核心功能: 
    1. 利用多模态大模型 (Rex-Omni/Qwen2.5-VL) 进行目标检测。
    2. 集成 SAHI (切片辅助推理) 解决小目标检测难题。
    3. 引入 Active Perception (主动感知) 机制，利用轨迹反馈指导下一帧切片。
    4. 集成 SORT 算法进行多目标关联与跟踪。
    5. 包含完整的调试工具：切片保存、双重可视化、显存清理。
"""

import torch
from PIL import Image, ImageFilter, ImageDraw, ImageFont  # ImageFilter 用于边缘增强
from rex_omni import RexOmniVisualize, RexOmniWrapper

import numpy as np
import os
import re
import sys
import gc  

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

# 确保 SORT.py 在路径下
try:
    from SORT import Sort, TrackingVisualizer
    SORT_AVAILABLE = True
except ImportError:
    print("Warning: SORT module not found. Tracking functionality will be disabled.")
    Sort = None
    TrackingVisualizer = None
    SORT_AVAILABLE = False

# ==================== 1. 核心工具函数：切片与图像处理 ====================

def get_image_slices(image, slice_h=640, slice_w=640, overlap_ratio=0.2, target_size=None):
    """
    标准网格切片函数 (Global Scan)
    
    原理：
    将高分辨率大图切割成多个重叠的小图（切片）。
    这解决了两个问题：
    1. 显存限制：避免一次性将 4K/8K 图送入显卡。
    2. 小目标检测：小目标在全图中可能只占几像素，切片后像素占比提升，特征更明显。
    
    参数:
        image: PIL Image 对象
        slice_h, slice_w: 物理切片的尺寸 (在原图上切多大)
        overlap_ratio: 重叠率，防止目标恰好被切在边缘
        target_size: [新增] 如果指定此参数，会将切片 Resize 到该尺寸 (超分放大)
    
    返回:
        slices: 列表，每个元素包含 [crop_image, x_offset, y_offset, (scale_x, scale_y)]
    """
    img_w, img_h = image.size
    slices = []
    
    # 计算步长 (Stride)
    stride_w = int(slice_w * (1 - overlap_ratio))
    stride_h = int(slice_h * (1 - overlap_ratio))
    
    # 生成切片的左上角坐标列表
    y_range = list(range(0, img_h - slice_h + stride_h, stride_h))
    x_range = list(range(0, img_w - slice_w + stride_w, stride_w))
    
    # 边界处理：确保最后一行/一列能覆盖图像边缘
    if img_h - y_range[-1] < slice_h: y_range[-1] = img_h - slice_h
    if img_w - x_range[-1] < slice_w: x_range[-1] = img_w - slice_w
        
    for y in y_range:
        for x in x_range:
            x = max(0, x)
            y = max(0, y)
            box = (x, y, x + slice_w, y + slice_h)
            # 执行裁剪
            crop = image.crop(box)
            
            # 超分逻辑 (Super-Resolution Strategy)
            # 如果物理切片较小 (如640)，但模型输入推荐 896，
            # 我们在这里强行放大切片，相当于给模型加了“数字变焦”。
            scale_x, scale_y = 1.0, 1.0
            if target_size is not None and (slice_w < target_size or slice_h < target_size):
                crop = crop.resize((target_size, target_size), Image.BILINEAR)
                scale_x = target_size / slice_w
                scale_y = target_size / slice_h
            
            # 边缘增强 (Edge Enhancement)
            # 对所有切片应用轻微锐化，帮助模型看清模糊的小目标轮廓
            crop = crop.filter(ImageFilter.EDGE_ENHANCE)
            
            # 强制加载数据到内存 (解决 PIL Lazy Loading 导致的 Batch 黑图问题)
            crop.load()
            
            # 保存切片信息：图片，偏移量，缩放比例
            slices.append([crop, x, y, (scale_x, scale_y)])
            
    return slices


def get_adaptive_slices(image, roi, target_size=800):
    """
    [策略 A: 主动感知切片] (Active Perception Slicing)
    
    功能描述：
        不进行网格扫描，而是根据给定的感兴趣区域 (ROI) 直接裁剪。
        通常用于利用上一帧的跟踪结果来“聚焦”当前帧的目标区域。
        
    参数详解:
        image: 原始图像。
        roi: 关注区域元组 (min_x, min_y, max_x, max_y)。
        target_size: 目标输入尺寸。会将 ROI 强制缩放(通常是放大)到此尺寸。
    """
    slices = []
    rx1, ry1, rx2, ry2 = roi
    roi_w = rx2 - rx1
    roi_h = ry2 - ry1

    # 裁剪
    crop = image.crop((rx1, ry1, rx2, ry2))
    
    # 计算缩放比例
    scale = 1.0
    if roi_w < target_size or roi_h < target_size:
        # 非等比缩放至 target_size
        crop = crop.resize((target_size, target_size), Image.BILINEAR)
        scale_x = target_size / roi_w
        scale_y = target_size / roi_h
        scale = (scale_x, scale_y) # 存储 tuple
    else:
        scale = (1.0, 1.0)
    
    # 边缘增强
    crop = crop.filter(ImageFilter.EDGE_ENHANCE)
    crop.load()

    # 返回格式: [image, offset_x, offset_y, scale_factor]
    slices.append([crop, rx1, ry1, scale])
    
    return slices

# ==================== 2. 核心工具函数：后处理算法 ====================

def nms(boxes, scores, iou_threshold=0.5):
    """
    [非极大值抑制] (Non-Maximum Suppression)
    
    功能描述：
        合并全图检测和切片检测产生的重复框。
        当两个框的重叠度 (IoU) 超过阈值时，只保留置信度高的那个。
    """
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)
    
    # 提取坐标
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # 计算每个框的面积
    areas = (x2 - x1) * (y2 - y1)
    
    # 按置信度从高到低排序，获取索引
    order = scores.argsort()[::-1]
    
    keep = []
    
    while order.size > 0:
        # 1. 保留当前分数最高的框
        i = order[0]
        keep.append(i)
        
        # 2. 计算当前框与剩余所有框的交集 (Intersection)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        # 3. 计算 IoU (Intersection over Union)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        # 4. 只保留 IoU 小于阈值的框 (即去除重叠严重的框)
        inds = np.where(iou <= iou_threshold)[0]
        
        # 更新 order，继续下一轮循环
        order = order[inds + 1]
        
    return keep

# ==================== 优化后的 ROI 计算函数 ====================

def calculate_smart_rois(tracked_objects, img_w, img_h, margin=200, max_area_ratio=0.6):
    """
    [主动感知核心 - 升级版] 计算下一帧关注区域 (Smart ROI Clustering)
    
    原理：
        如果上一帧跟踪到了物体，下一帧物体大概率还在附近。
        升级点：如果多个物体距离较远，不再强行合并成一个大框（因为那样会导致 ROI 
        过大从而失效），而是智能分裂成 1-2 个独立的 ROI。
        
    参数:
        tracked_objects: 跟踪列表 [x1, y1, x2, y2, id]
        img_w, img_h: 图像宽高
        margin: 向外扩的边距
        max_area_ratio: 单个 ROI 允许的最大面积占比，超过则认为不如全图扫描
        
    返回:
        roi_list: 包含一个或多个 ROI 元组 [(x1, y1, x2, y2), ...]
    """
    if len(tracked_objects) == 0:
        return []

    # 提取所有轨迹的中心点，用于判断距离
    centers = []
    for obj in tracked_objects:
        cx = (obj[0] + obj[2]) / 2
        cy = (obj[1] + obj[3]) / 2
        centers.append([cx, cy, obj]) # 把原始 obj 也存进去

    # 简单聚类逻辑：
    # 如果目标数量 > 1，计算最远两个目标的距离。
    # 如果距离超过图像宽度的 1/2，则尝试分裂成两组。
    
    groups = []
    if len(centers) > 1:
        centers.sort(key=lambda x: x[0]) # 按 x 坐标排序
        
        # 计算首尾距离
        dist_x = centers[-1][0] - centers[0][0]
        
        # 如果横向跨度太大，就切成左右两组
        if dist_x > (img_w * 0.5):
            mid = len(centers) // 2
            groups.append([c[2] for c in centers[:mid]]) # 左边一组
            groups.append([c[2] for c in centers[mid:]]) # 右边一组
        else:
            groups.append([c[2] for c in centers]) # 只有一组
    else:
        groups.append([tracked_objects[0]])

    final_rois = []
    
    for group in groups:
        group_np = np.array(group)
        x1s, y1s = group_np[:, 0], group_np[:, 1]
        x2s, y2s = group_np[:, 2], group_np[:, 3]

        # 找到包含该组目标的框
        min_x = np.min(x1s)
        min_y = np.min(y1s)
        max_x = np.max(x2s)
        max_y = np.max(y2s)

        # 加上边距 (Margin)，防止目标跑出视野
        min_x = max(0, int(min_x - margin))
        min_y = max(0, int(min_y - margin))
        max_x = min(img_w, int(max_x + margin))
        max_y = min(img_h, int(max_y + margin))

        # 校验：如果区域太小，或者太大，则忽略
        roi_w = max_x - min_x
        roi_h = max_y - min_y
        area_ratio = (roi_w * roi_h) / (img_w * img_h)
        
        if roi_w < 50 or roi_h < 50:
            continue # 太小忽略
            
        if area_ratio > max_area_ratio:
            # 如果单个 ROI 占比已经很大，说明目标太散，不仅不需要切片，
            # 可能直接全图 resize 效果更好，或者这一帧应该触发 Global Rescan
            continue 

        final_rois.append((min_x, min_y, max_x, max_y))

    return final_rois


# ==================== 3. 主程序 ====================

def main():
    # ==================== 配置参数 ====================
    image_folder = "/home/REXOMNI-125-main/Rex-Omni/Dataset/vis2019-mot-test-dev-mini/uav0000297_02761_v"

    # 自动解析序列信息
    img_basename = os.path.basename(os.path.normpath(image_folder)) 
    # 查找uav在字符串中的位置
    uav_index = img_basename.find('uav')
    if uav_index != -1 and len(img_basename) >= uav_index + 10:
        suffix = img_basename[uav_index + 7:uav_index + 10]  
    else:
        suffix = "001"  # 默认值
    sequence_name = os.path.basename(os.path.normpath(image_folder))

    # 获取Rex-Omni根目录
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


    categories = ["car", "tiny car", "distant vehicle"]
    CLASS_ID_MAP = {"car": 1, "tiny car": 1, "distant vehicle": 1, "vehicle": 1}
    
    tracking_results = []
    
    MAX_INFERENCE_BATCH_SIZE = 4 

    # 限制单次推理的最大切片数。如果切片总数超过此值，将分批推理。
    # ==================== 加载模型 ====================
    model_path = "/home/weights"
    rex_model = RexOmniWrapper(
        model_path=model_path,
        backend="transformers",
        max_tokens=4096,
        # temperature=0.0,
        # top_p=0.05,
        # top_k=1,
        repetition_penalty=1.05,
    )
    
    # ---------------- Tracker 初始化 ----------------
    if SORT_AVAILABLE:
        print("Initializing SORT tracker...")
        # max_age=30: 目标消失 30 帧内仍保留 ID (抗遮挡)
        # min_hits=5: 目标连续出现 3 帧才确认为有效轨迹 (抗噪声)
        tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.1)
        visualizer = TrackingVisualizer()
    else:
        print("Error: SORT module is missing. Exiting.")
        return
    
    # ---------------- 图片读取 ----------------
    image_files = sorted(
        [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".png"))],
        # 智能排序：按文件名中的数字排序，避免 frame_10 排在 frame_2 前面
        key=lambda x: int(re.search(r"\d+", x).group()) if re.search(r"\d+", x) else x,   
    )
    if not image_files: raise ValueError(f"No images found in {image_folder}")

    #  调试模式：只跑前10帧
    #  DEBUG_FRAME_LIMIT = 200
    #  image_files = image_files[:DEBUG_FRAME_LIMIT]

    print(f"[DEBUG MODE] Only processing first {len(image_files)} frames.")
    # print(f"[DEBUG MODE] Slices will be saved to: {DEBUG_SLICES_DIR}")

  


    # ---------------- 切片策略参数 ----------------
    SLICE_SIZE = 640            # 物理切片大小 (越小越能聚焦局部，但也越容易切碎大目标)
    SLICE_UPSAMPLE_SIZE = 896   # 送入模型的逻辑大小 (大模型喜欢大图，建议 800-1000)
    SLICE_OVERLAP = 0.2         # 重叠率
    CONF_THRESHOLD = 0.25       # 置信度阈值 (设低一点，让小目标能进来，靠后续 NMS 过滤)
    
    # 强制全扫描的周期 (防止隧道视野)
    # 每 30 帧强制进行一次全图扫描，防止有新目标从画面边缘进入而被主动模式漏掉
    GLOBAL_RESCAN_INTERVAL = 30 

    # 热身机制：前 5 帧强制全图网格扫描，确保卡尔曼滤波在切换 ROI 模式前已经获得稳定初值
    GLOBAL_WARMUP_FRAMES = 5

    # 状态变量：存储下一帧的关注区域
    next_frame_roi = None 
    tracking_results = []

    print(f"共找到 {len(image_files)} 帧图片 (Debug Mode)")
    print(f"序列名称: {sequence_name}")
    print(f"切片策略: 物理尺寸={SLICE_SIZE}, 放大尺寸={SLICE_UPSAMPLE_SIZE}, 重叠率={SLICE_OVERLAP}")
    

    # ==================== 逐帧处理 ==================== 
    for frame_idx, image_file in enumerate(image_files, 1):
        print(f"\r处理帧: {frame_idx}/{len(image_files)}", end="", flush=True)
        
        # 加载全图并强制 load
        image_path = os.path.join(image_folder, image_file)
        full_image = Image.open(image_path).convert("RGB")
        full_image.load() 
        img_w, img_h = full_image.size

        # ---------------- A. 智能切片决策逻辑修改 ----------------
        slices_info = []
        slice_mode = "Global"
        use_active = False
        
        # 1. 强制热身期 
        if frame_idx <= GLOBAL_WARMUP_FRAMES:
            use_active = False
            slice_mode = "Global_Warmup"
        
        # 2. 周期性全场重扫描
        elif frame_idx % GLOBAL_RESCAN_INTERVAL == 0:
            use_active = False
            slice_mode = "Global_Rescan"
        
        # 3. 主动感知判断 (现在支持 next_frame_rois 列表)
        elif next_frame_rois and len(next_frame_rois) > 0:
            use_active = True
            slice_mode = f"Active_Multi_{len(next_frame_rois)}"
        else:
            use_active = False
            slice_mode = "Global_Lost"

        # 执行物理切片
        if use_active:
            # 遍历所有的 ROI 生成切片
            for roi in next_frame_rois:
                # 复用 get_adaptive_slices，它返回的是一个 list，我们 extend 进去
                roi_slices = get_adaptive_slices(full_image, roi, target_size=SLICE_UPSAMPLE_SIZE)
                slices_info.extend(roi_slices)
        else:
            slices_info = get_image_slices(full_image, slice_h=SLICE_SIZE, slice_w=SLICE_SIZE, 
                                           overlap_ratio=SLICE_OVERLAP, target_size=SLICE_UPSAMPLE_SIZE)


        # -----------  步骤 B: 拆分推理 (Split Inference) ----------------
        # 关键优化：全图和切片分开推理，避免 Padding 导致的精度下降
        
        # 1. 全图推理 (始终执行，作为兜底)
        full_img_results = rex_model.inference(
            images=[full_image], 
            task="mot_detection", 
            categories=categories
        )
        
        # 2. 切片推理
        slice_imgs = [s[0] for s in slices_info]
        slice_results = []
        #  切片分批推理 (防止 OOM)
        if slice_imgs:
            for i in range(0, len(slice_imgs), MAX_INFERENCE_BATCH_SIZE):
                batch = slice_imgs[i : i + MAX_INFERENCE_BATCH_SIZE]
                # 推理完成后立即返回结果，不积压中间变量
                batch_res = rex_model.inference(
                images=batch, 
                task="mot_detection", 
                categories=categories
                )
                slice_results.extend(batch_res)
                # 每一批次处理完后，可以视情况手动清理
                torch.cuda.empty_cache()   
        
            
        # ---------------- 步骤 C: 结果合并与解析 ----------------
        # 构造合并队列，统一格式: (result_dict, [offset_x, offset_y], scale_factor, is_slice)
        processing_queue = []
        # 将全图检测加入队列
        if full_img_results and full_img_results[0]["success"]:
            processing_queue.append((full_img_results[0], [0, 0], (1.0, 1.0), False))
        # 将所有切片检测加入队列
        for res, info in zip(slice_results, slices_info):
            if res["success"]:
                # info: [crop, x, y, scale]
                scale = info[3]
                # 兼容 scale 是 float 还是 tuple
                if isinstance(scale, float): scale = (scale, scale)
                processing_queue.append((res, [info[1], info[2]], scale, True))

        all_detections = [] 

        # 边缘触碰判定阈值 
        EDGE_THRESHOLD = 5

        # 4. 统一解析循环
        for result, offset, scale, is_slice in processing_queue:
            predictions = result.get("extracted_predictions", {})
            confidence = result.get("token_confidences", [])
            offset_x, offset_y = offset
            scale_x, scale_y = scale
            
            i = 4 
            for cat, items in predictions.items():
                for item in items:
                    if item["type"] == "box":
                        bbox = item['coords']  # [x1, y1, x2, y2]

                        # --- [策略 1: 巨型幻觉框过滤] ---
                        # 计算检测框在当前切片/图像中的面积占比
                        box_w_logic = bbox[2] - bbox[0]
                        box_h_logic = bbox[3] - bbox[1]
                        # 逻辑空间总面积 (对于切片通常是 896*896)
                        current_total_area = SLICE_UPSAMPLE_SIZE * SLICE_UPSAMPLE_SIZE if is_slice else (img_w * img_h)
                        area_ratio = (box_w_logic * box_h_logic) / current_total_area
                        
                        if is_slice:
                            if "Global" in slice_mode:
                                # 全局盲扫模式下，单个目标不应占满切片 (阈值 0.6)
                                if area_ratio > 0.6: 
                                    i += 5; continue
                            else:
                                # 主动感知模式下，允许目标占满，但不能是 100% 的背景误报 (阈值 0.95)
                                if area_ratio > 0.95: 
                                    i += 5; continue

                        # --- [策略 2: 边缘截断过滤 (解决“框一半”问题)] ---
                        if is_slice:
                            # 判定是否触碰了切片的逻辑边缘
                            touches_left   = bbox[0] <= EDGE_THRESHOLD
                            touches_top    = bbox[1] <= EDGE_THRESHOLD
                            touches_right  = bbox[2] >= (SLICE_UPSAMPLE_SIZE - EDGE_THRESHOLD)
                            touches_bottom = bbox[3] >= (SLICE_UPSAMPLE_SIZE - EDGE_THRESHOLD)

                            # 如果触碰了切片边缘，且该边缘不是原图的物理边缘，则视为截断框，直接舍弃
                            if (touches_left and offset_x > 0) or \
                               (touches_top and offset_y > 0) or \
                               (touches_right and (offset_x + SLICE_SIZE < img_w - 5)) or \
                               (touches_bottom and (offset_y + SLICE_SIZE < img_h - 5)):
                                i += 5; continue

                        # --- [坐标还原逻辑] ---
                        # 原理: (模型输出坐标 / 缩放系数) + 切片左上角偏移
                        real_x1 = (bbox[0] / scale_x) + offset_x
                        real_y1 = (bbox[1] / scale_y) + offset_y
                        real_x2 = (bbox[2] / scale_x) + offset_x
                        real_y2 = (bbox[3] / scale_y) + offset_y
                        
                        # --- [策略 3: 宽高比与尺寸过滤 (解决“窄长框”问题)] ---
                        bw, bh = real_x2 - real_x1, real_y2 - real_y1
                        
                        # A. 基础尺寸过滤：过滤掉物理像素过小的噪点
                        if bw < 10 or bh < 10: 
                            i += 5; continue
                            
                        # B. 宽高比过滤：过滤比例畸形（极窄或极扁）的框
                        if bh > 0:
                            aspect_ratio = bw / bh
                            # 设置正常车辆的比例范围，例如 0.25 到 4.0
                            if aspect_ratio > 4.0 or aspect_ratio < 0.25:
                                i += 5; continue

                        # 坐标约束：确保不会超出原图边界
                        real_x1, real_y1 = max(0, real_x1), max(0, real_y1)
                        real_x2, real_y2 = min(img_w, real_x2), min(img_h, real_y2)

                        # --- [置信度计算与收集] ---
                        # 取 4 个坐标点 token 的平均置信度
                        score = sum(confidence[i:i+4])/4 if i+4 < len(confidence) else 0.5
                        i += 5 # 指针跳向下一个预测目标

                        # 最终置信度门槛过滤
                        if score > CONF_THRESHOLD:
                            all_detections.append([real_x1, real_y1, real_x2, real_y2, score])


        # ---------------- 步骤 D: NMS 与 跟踪更新 ----------------
        final_detections = np.empty((0, 5))
        if len(all_detections) > 0:
            all_dets_np = np.array(all_detections)
            boxes = all_dets_np[:, :4]
            scores = all_dets_np[:, 4]
            keep_indices = nms(boxes, scores, iou_threshold=0.25)
            final_detections = all_dets_np[keep_indices]
      
        #  更新追踪器
        tracked_objects = tracker.update(final_detections)
        
        # 计算下一帧的关注区域 (Feedback Loop)
        # 使用新的 Smart Clustering 函数，返回列表
        next_frame_rois = calculate_smart_rois(tracked_objects, img_w, img_h, margin=200)

        # ---------------- 步骤 E: 结果保存与可视化----------------
        # 收集 TXT 结果
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = obj
            w, h = x2 - x1, y2 - y1
            tracking_results.append([
                frame_idx, int(track_id), x1, y1, w, h, 1, 1, 1
            ])
        
        # 字体加载 (尝试加载系统字体，失败则用默认)
        try:
            # 增大字号方便查看
            font = ImageFont.truetype("arial.ttf", 30)
            small_font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = None
            small_font = None

        # --- 左图绘制：Rex-Omni 原始检测 ---
        vis_det = full_image.copy()
        draw_det = ImageDraw.Draw(vis_det)
        
        # 绘制所有原始检测框 (红色)
        for det in final_detections:
            # 如果 det 是字典 
            if isinstance(det, dict):
                dx1, dy1, dx2, dy2 = det['box_2d']
                dscore = det['score']
            else:
                # 如果 det 是列表 
                dx1, dy1, dx2, dy2, dscore = det[:5]

            # 画红框
            draw_det.rectangle([dx1, dy1, dx2, dy2], outline="red", width=3)
            # 标上分数 
            draw_det.text((dx1, max(0, dy1-20)), f"{dscore:.2f}", fill="red", font=small_font)


        # 绘制 Active ROI (黄色 - 适配多 ROI 列表)
        if use_active and next_frame_rois:
            for i, roi in enumerate(next_frame_rois):
                rx1, ry1, rx2, ry2 = roi
                # 画黄色框表示模型聚焦区域
                draw_det.rectangle([rx1, ry1, rx2, ry2], outline="yellow", width=5)
                # 标记
                draw_det.text((rx1 + 5, ry1 + 5), f"Active ROI #{i+1}", fill="yellow", font=small_font) 
        
        # 加上标题 "Raw Detection"
        draw_det.text((20, 20), "Raw Detection (Red)", fill="red", font=font)

        # 保存检测图片
        VIS_OUTPUT_DIR_1 = os.path.join(VIS_OUTPUT_DIR, "det")
        os.makedirs(VIS_OUTPUT_DIR_1, exist_ok=True)
        vis_det.save(os.path.join(VIS_OUTPUT_DIR_1, f"frame_{frame_idx:03d}.jpg"))

        # --- 右图绘制：SORT 跟踪结果 ---
        # 使用 Visualizer 画出带 ID 的彩框
        vis_track = visualizer.draw_tracks(
            image=full_image,
            tracks=tracked_objects,
            font_size=20,
            line_width=3,
            show_trajectory=True
        )
        draw_track = ImageDraw.Draw(vis_track)
        
        # 同样把 ROI 画上去，方便左右对照
        if use_active and next_frame_rois:
            for roi in next_frame_rois:
                draw_track.rectangle(roi, outline="yellow", width=4)
            
        # 加上标题 "SORT Tracking"
        draw_track.text((20, 20), "SORT Tracking (ID)", fill="green", font=font)

        # 保存跟踪图片
        VIS_OUTPUT_DIR_2 = os.path.join(VIS_OUTPUT_DIR, "track")
        os.makedirs(VIS_OUTPUT_DIR_2, exist_ok=True)
        vis_track.save(os.path.join(VIS_OUTPUT_DIR_2, f"frame_{frame_idx:03d}.jpg"))


        # --- 拼图 ---
        # 准备画布: 左右双图拼接
        # 左图：Raw Detections (原始检测)
        # 右图：SORT Tracking (最终跟踪)
        combined_w = img_w * 2
        combined_h = img_h
        combined_image = Image.new('RGB', (combined_w, combined_h))

        combined_image.paste(vis_det, (0, 0))
        combined_image.paste(vis_track, (img_w, 0))
        
        # 中间画一条白线分割
        draw_comb = ImageDraw.Draw(combined_image)
        draw_comb.line([(img_w, 0), (img_w, combined_h)], fill="white", width=5)

        # 保存拼接后的大图
        VIS_OUTPUT_DIR_3 = os.path.join(VIS_OUTPUT_DIR, "combined")
        os.makedirs(VIS_OUTPUT_DIR_3, exist_ok=True)
        combined_image.save(os.path.join(VIS_OUTPUT_DIR_3, f"frame_{frame_idx:03d}.jpg"))

        # ---------------- 步骤 F: 显存清理 ----------------
        del full_image, slices_info, slice_imgs, full_img_results, slice_results, processing_queue
        del all_detections, final_detections, combined_image, vis_det, vis_track
        gc.collect()
        torch.cuda.empty_cache()
    
    # ==================== 保存评估文件 ====================
    result_file_path = os.path.join(SAVE_RESULTS_DIR, f"{sequence_name}.txt")
    with open(result_file_path, 'w+') as fid:
        for result in tracking_results:
            fid.write('%d,%d,%0.2f,%0.2f,%0.2f,%0.2f,%d,%d,%d\n' % tuple(result))
    
    print(f"\n 处理完成！")
    print(f"   - 结果文件: {result_file_path}")
    print(f"   - 可视化帧: {VIS_OUTPUT_DIR}/frame_*.jpg")
    # print(f"   - 调试切片: {DEBUG_SLICES_DIR}/frame_*/")
    print(f"   - 总记录数: {len(tracking_results)} 条")

if __name__ == "__main__":
    main()