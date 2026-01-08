import torch
import platform
import psutil
import sys

def get_env_info():
    print("="*20 + " Experimental Environment Report " + "="*20)
    
    # 1. 操作系统信息
    print(f"[OS]: {platform.system()} {platform.release()} ({platform.architecture()[0]})")
    
    # 2. Python 与 核心库版本
    print(f"[Python]: {sys.version.split()[0]}")
    print(f"[PyTorch]: {torch.__version__}")
    print(f"[CUDA Available]: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[CUDA Version]: {torch.version.cuda}")
    
    # 3. CPU 信息
    print(f"[CPU]: {platform.processor()}")
    print(f"[CPU Cores]: {psutil.cpu_count(logical=False)} Physical, {psutil.cpu_count(logical=True)} Logical")
    
    # 4. 内存信息
    mem = psutil.virtual_memory()
    print(f"[RAM]: {round(mem.total / (1024**3), 2)} GB")
    
    # 5. GPU 信息 (如果可用)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"[GPU {i}]: {props.name}")
            print(f"[VRAM {i}]: {round(props.total_memory / (1024**3), 2)} GB")
    else:
        print("[GPU]: No NVIDIA GPU detected.")
        
    print("="*60)

if __name__ == "__main__":
    get_env_info()