import torch

torch.cuda.is_available()  # 判断PyTorch是否支持GPU加速
torch.cuda.device_count()  # 返回GPU的数量
torch.cuda.current_device()  # 返回当前设备索引
torch.cuda.get_device_name(0)  # 返回当前设备信息
