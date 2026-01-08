import cv2
import os
from pathlib import Path

def images_to_video_opencv(
    image_folder: str,
    output_video: str,
    fps: int = 30,
    resize: tuple = None  # (width, height)
):
    """
    将文件夹中的图片序列转换为 MP4 视频
    
    参数:
        image_folder: 图片所在文件夹路径
        output_video: 输出视频文件路径（如 output.mp4）
        fps: 视频帧率
        resize: 可选，统一调整图片尺寸 (宽, 高)
    """
    # 获取所有图片文件，按文件名排序
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = sorted([
        f for f in Path(image_folder).iterdir()
        if f.suffix.lower() in valid_extensions
    ])
    
    if not image_files:
        raise ValueError(f"在 {image_folder} 中未找到有效图片文件")
    
    # 读取第一张图片获取尺寸信息
    first_frame = cv2.imread(str(image_files[0]))
    if first_frame is None:
        raise ValueError(f"无法读取图片: {image_files[0]}")
    
    height, width = first_frame.shape[:2]
    if resize:
        width, height = resize
    
    # 配置视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 编码
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        raise RuntimeError("无法创建视频写入器")
    
    # 逐帧写入视频
    for i, image_path in enumerate(image_files):
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"警告: 跳过无法读取的图片 {image_path}")
            continue
        
        # 尺寸调整（如果指定）
        if resize:
            frame = cv2.resize(frame, resize)
        
        video_writer.write(frame)
        
        # 进度提示
        if (i + 1) % 30 == 0:
            print(f"已处理: {i + 1}/{len(image_files)} 帧")
    
    video_writer.release()
    print(f"✓ 视频已保存至: {output_video}")

# 使用示例
if __name__ == "__main__":
    # 假设图片在 "frames" 文件夹，命名如 frame_001.png, frame_002.png...
    images_to_video_opencv(
        image_folder="/home/REXOMNI-125-main/Rex-Omni/output/sequence_3",
        output_video="/home/REXOMNI-125-main/Rex-Omni/output/output_3.mp4",
        fps=20,
        # resize=(1920, 1080)  # 可选：统一调整分辨率
    )