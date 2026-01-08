#!/usr/bin/env python3
import os
import sys
import glob
import argparse

def convert_visdrone_to_mot16(input_path, output_path, filter_classes=False):
    """
    将VisDrone格式的标注转换为标准MOT16格式
    
    VisDrone格式: id,frame,x,y,w,h,consider_flag,class,truncation,occlusion
    MOT16格式: frame,id,x,y,w,h,conf,class,visibility
    
    修改: frame从1开始计数，所有坐标值转为整数
    """
    try:
        converted_lines = []
        skip_count = 0
        
        with open(input_path, 'r') as fin:
            for line_num, line in enumerate(fin, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(',')
                
                if len(parts) != 10:
                    print(f"  警告: 第{line_num}行列数不正确（{len(parts)}列），跳过")
                    skip_count += 1
                    continue
                
                try:
                    # 解析VisDrone格式并确保所有值为整数
                    id_ = int(float(parts[0]))
                    frame = int(float(parts[1])) + 1  # 从1开始计数
                    x = int(float(parts[2]))          # 确保为整数
                    y = int(float(parts[3]))
                    w = int(float(parts[4]))
                    h = int(float(parts[5]))
                    consider_flag = int(float(parts[6]))
                    class_id = int(float(parts[7]))
                    truncation = int(float(parts[8]))
                    occlusion = int(float(parts[9]))
                    
                    # 过滤器类别（可选）
                    if filter_classes and class_id not in [4, 5, 6, 9]:
                        continue
                    
                    # 计算visibility（可见度）
                    if occlusion == 0 and truncation == 0:
                        visibility = 1.0
                    elif occlusion == 2:
                        visibility = 0.33
                    else:
                        visibility = 0.66
                    
                    # 构建MOT16格式行（frame和id已交换位置）
                    mot_line = [
                        str(frame),           # frame（从1开始）
                        str(id_),             # id
                        str(x),               # x（整数）
                        str(y),               # y（整数）
                        str(w),               # width（整数）
                        str(h),               # height（整数）
                        str(consider_flag),   # confidence
                        str(class_id),        # class
                        f"{visibility:.2f}"   # visibility
                    ]
                    
                    converted_lines.append(','.join(mot_line))
                    
                except ValueError as e:
                    print(f"  警告: 第{line_num}行数据格式错误: {e}")
                    skip_count += 1
                    continue
        
        # 写入输出文件
        with open(output_path, 'w') as fout:
            fout.write('\n'.join(converted_lines))
        
        return True, len(converted_lines), skip_count
        
    except Exception as e:
        print(f"  错误: {e}")
        return False, 0, 0

def main():
    parser = argparse.ArgumentParser(
        description='将VisDrone格式转换为MOT16格式（frame从1开始，坐标值为整数）',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--input', '-i', 
                        default='/data/lz/pythonProjects/REXOMNI-125/Rex-Omni/dataset/annotations/uav0000326_01035_v.txt',
                        help='输入文件或目录路径')
    parser.add_argument('--output', '-o', 
                        default='/data/lz/pythonProjects/REXOMNI-125/Rex-Omni/dataset/annotations/gt_326.txt',
                        help='输出文件或目录路径')
    parser.add_argument('--ext', default='.txt', 
                        help='文件扩展名（处理目录时使用，默认: .txt）')
    parser.add_argument('--filter-classes', action='store_true', 
                        help='只保留车辆类别（4=汽车,5=面包车,6=卡车,9=公共汽车）')
    parser.add_argument('--dry-run', action='store_true', 
                        help='只显示转换信息，不实际写入文件')
    
    args = parser.parse_args()
    
    # 统计信息
    total_files = 0
    success_files = 0
    total_lines = 0
    total_skipped = 0
    
    # 检查输入类型
    if os.path.isfile(args.input):
        # 处理单个文件
        total_files = 1
        if not args.dry_run:
            os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
            success, lines, skipped = convert_visdrone_to_mot16(args.input, args.output, args.filter_classes)
            if success:
                success_files = 1
                total_lines = lines
                total_skipped = skipped
                print(f"✓ 转换成功: {args.input} -> {args.output}")
                print(f"  {lines} 行已转换, {skipped} 行被跳过")
            else:
                print(f"✗ 转换失败: {args.input}")
                sys.exit(1)
        else:
            print(f"将转换: {args.input} -> {args.output}")
    
    elif os.path.isdir(args.input):
        # 处理目录
        pattern = os.path.join(args.input, f'*{args.ext}')
        input_files = sorted(glob.glob(pattern))
        
        if not input_files:
            print(f"警告: 在 {args.input} 中未找到 {args.ext} 文件")
            sys.exit(0)
        
        total_files = len(input_files)
        print(f"找到 {total_files} 个文件需要转换\n")
        
        # 创建输出目录
        if not args.dry_run:
            os.makedirs(args.output, exist_ok=True)
        
        for i, input_file in enumerate(input_files, 1):
            filename = os.path.basename(input_file)
            output_file = os.path.join(args.output, filename)
            
            print(f"[{i}/{total_files}] {filename}...")
            
            if not args.dry_run:
                success, lines, skipped = convert_visdrone_to_mot16(input_file, output_file, args.filter_classes)
                if success:
                    success_files += 1
                    total_lines += lines
                    total_skipped += skipped
                    print(f"  ✓ {lines} 行已转换, {skipped} 行被跳过")
                else:
                    print(f"  ✗ 失败")
        
        print(f"\n{'='*50}")
        print(f"处理完成: {success_files}/{total_files} 个文件成功")
        print(f"总行数: {total_lines}")
        print(f"跳过行数: {total_skipped}")
        if not args.dry_run:
            print(f"输出目录: {args.output}")
    else:
        print(f"错误: 输入路径 {args.input} 不存在")
        sys.exit(1)
    
    if args.filter_classes:
        print("\n⚠️  注意: 您启用了类别过滤，只保留车辆类别（4,5,6,9）")

if __name__ == '__main__':
    main()