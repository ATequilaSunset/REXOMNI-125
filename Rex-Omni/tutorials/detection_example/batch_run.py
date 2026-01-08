import subprocess
import os
import time

# 1. 定义你想要依次处理的路径列表
image_folders = [

    '/home/REXOMNI-125-main/Rex-Omni/Dataset/vis2019-mot-test-dev-mini/uav0000201_00000_v' ,
    '/home/REXOMNI-125-main/Rex-Omni/Dataset/vis2019-mot-test-dev-mini/uav0000249_02688_v' ,
    '/home/REXOMNI-125-main/Rex-Omni/Dataset/vis2019-mot-test-dev-mini/uav0000297_00000_v' ,
    '/home/REXOMNI-125-main/Rex-Omni/Dataset/vis2019-mot-test-dev-mini/uav0000306_00230_v' ,
    '/home/REXOMNI-125-main/Rex-Omni/Dataset/vis2019-mot-test-dev-mini/uav0000355_00001_v' ,
    '/home/REXOMNI-125-main/Rex-Omni/Dataset/vis2019-mot-test-dev-mini/uav0000370_00001_v' 
]

# 2. 指定主脚本名称
script_name = "/home/REXOMNI-125-main/Rex-Omni/tutorials/detection_example/detection_example_SAHI_work.py"

def run_experiments():
    for i, folder in enumerate(image_folders):
        print(f"\n[任务 {i+1}/{len(image_folders)}] 正在启动: {os.path.basename(folder)}")
        
        # 构造命令行指令
        # 如果需要指定 GPU，可以在前面加上 CUDA_VISIBLE_DEVICES=0
        command = [
            "python", script_name,
            "--image_folder", folder
        ]
        
        start_time = time.time()
        
        try:
            # 运行脚本并实时打印输出
            result = subprocess.run(command, check=True)
            
            end_time = time.time()
            duration = (end_time - start_time) / 60
            print(f"处理完成: {os.path.basename(folder)} | 耗时: {duration:.2f} 分钟")
            
        except subprocess.CalledProcessError as e:
            print(f"运行失败: {os.path.basename(folder)}，错误代码: {e.returncode}")
            continue # 出错后继续运行下一个

    print("\n🎉 所有任务处理完毕！")

if __name__ == "__main__":
    run_experiments()