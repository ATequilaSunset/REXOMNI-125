from PIL import Image, ImageFilter
import numpy as np
import cv2

def sharpen_image_pil(image, method='unsharp_mask', strength=2):
    """
    PIL图像锐化处理
    
    参数:
        image: PIL图像对象
        method: 锐化方法 ('sharpen', 'unsharp_mask', 'custom')
        strength: 锐化强度 (仅对custom有效)
    """
    if method == 'sharpen':
        # 基础锐化滤镜 (效果较弱)
        sharpened = image.filter(ImageFilter.SHARPEN)
        
    elif method == 'unsharp_mask':
        # 非锐化掩码 (推荐，效果更自然)
        # radius=2, percent=150, threshold=3 是常用参数
        sharpened = image.filter(
            ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)
        )
        
    elif method == 'custom':
        # 自定义卷积核锐化 (强度可调)
        from PIL import ImageEnhance
        
        # 使用ImageFilter.Kernel定义锐化核
        # 3x3锐化核，中心为5，周围为-1
        sharpen_kernel = ImageFilter.Kernel(
            size=(3, 3),
            kernel=[-1, -1, -1,
                    -1,  9, -1,
                    -1, -1, -1],
            scale=1,
            offset=0
        )
        sharpened = image.filter(sharpen_kernel)
        
        # 进一步增强
        enhancer = ImageEnhance.Sharpness(sharpened)
        sharpened = enhancer.enhance(strength)
    
    return sharpened

def resize_sharpen_and_save(image_path, output_path, 
                            target_size=(1024, 1024),
                            sharpen_method='unsharp_mask'):
    """
    完整流程：打开图像 → Lanczos插值resize → 锐化 → 保存
    
    参数:
        image_path: 输入图像路径
        output_path: 输出图像路径
        target_size: 目标尺寸 (宽, 高)
        sharpen_method: 锐化方法
    """
    # 1. 打开并resize
    image = Image.open(image_path).convert("RGB")
    print(f"原始尺寸: {image.size}")
    
    resized_image = image.resize(
        target_size, 
        Image.Resampling.LANCZOS
    )
    print(f"Resize后尺寸: {resized_image.size}")
    
    # 2. 锐化处理
    sharpened_image = sharpen_image_pil(resized_image, method=sharpen_method)
    print(f"锐化方法: {sharpen_method}")
    
    # 3. 保存结果
    sharpened_image.save(output_path, quality=95)
    print(f"已保存锐化后的图像: {output_path}")
    
    return sharpened_image

# 使用示例
if __name__ == "__main__":
    input_img = "/home/REXOMNI-125-main/Rex-Omni/dataset/visdrone2019/264/sahi_for_test/slice_1025_0_1281_256.png"
    output_img = "/home/REXOMNI-125-main/Rex-Omni/dataset/visdrone2019/264/sahi_for_test/slice_1024_1024_sharpen.png"
    
    # 执行完整流程
    result = resize_sharpen_and_save(
        input_img, 
        output_img,
        sharpen_method='unsharp_mask'  # 推荐
    )