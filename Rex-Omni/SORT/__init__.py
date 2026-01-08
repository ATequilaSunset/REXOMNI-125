# SORT/__init__.py
"""
SORT (Simple, Online and Realtime Tracker) 模块

这是一个简单、在线且实时的多目标跟踪实现。

核心功能:
- Sort: 主跟踪器类，用于多目标跟踪
- KalmanBoxTracker: 单个目标的卡尔曼滤波跟踪器

辅助功能:
- iou_batch: 计算批次IoU
- linear_assignment: 解决分配问题
- convert_dets_for_sort: 将检测结果转换为SORT格式 (如果依赖可用)

基本用法:
    >>> from SORT import Sort
    >>> tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
    >>> detections = np.array([[100, 50, 200, 150, 0.9], [300, 200, 400, 300, 0.85]])
    >>> tracked_objects = tracker.update(detections)
"""

from .sort import (
    Sort,
    TrackingVisualizer,
    KalmanBoxTracker,
    linear_assignment,
    iou_batch,
    convert_bbox_to_z,
    convert_x_to_bbox,
    associate_detections_to_trackers
)

# 尝试导入辅助函数，但如果依赖缺失则静默失败
try:
    from .testTrackingSort import (
        pre_process,
        convert_dets_for_sort,
        test
    )
    _TEST_AVAILABLE = True
except ImportError:
    _TEST_AVAILABLE = False

__version__ = "1.0.0"

__all__ = [
    # 核心类
    'Sort',
    'KalmanBoxTracker',
    'TrackingVisualizer'
    # 核心工具函数
    'linear_assignment',
    'iou_batch',
    'convert_bbox_to_z',
    'convert_x_to_bbox',
    'associate_detections_to_trackers',
]

# 如果辅助函数可用，添加到__all__
if _TEST_AVAILABLE:
    __all__.extend([
        'pre_process',
        'convert_dets_for_sort',
        'test'
    ])

def create_trackers(num_classes, max_age=30, min_hits=3, iou_threshold=0.3):
    """
    为多个类别创建跟踪器字典
    
    Args:
        num_classes: 类别总数
        max_age: 最大跟踪帧数
        min_hits: 最小命中次数
        iou_threshold: IoU匹配阈值
        
    Returns:
        dict: {类别ID: Sort实例}
    """
    return {
        cls_id: Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
        for cls_id in range(1, num_classes + 1)
    }

# 自检函数
def check_installation():
    """检查SORT模块安装状态"""
    print(f"SORT模块版本: {__version__}")
    print(f"核心组件: {'✓' if 'Sort' in globals() else '✗'} Sort, {'✓' if 'KalmanBoxTracker' in globals() else '✗'} KalmanBoxTracker")
    print(f"辅助组件: {'✓' if _TEST_AVAILABLE else '✗'} testTrackingSort")
    if not _TEST_AVAILABLE:
        print("提示: 如需使用 testTrackingSort 的辅助函数，请确保相关依赖已安装 (torch, numpy, skimage等)")