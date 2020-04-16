import time
import torch


flag = torch.cuda.is_available()
print(flag)
ngpu= 1
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)
print(torch.cuda.get_device_name(0))
print(torch.rand(3,3).cuda())
print(torch.__version__)        # 返回pytorch的版本
a = torch.randn(10000, 10000)    # 返回10000行1000列的张量矩阵
b = torch.randn(10000, 20000)     # 返回1000行2000列的张量矩阵
t0 = time.time()        # 记录时间
c = torch.matmul(a, b)      # 矩阵乘法运算
t1 = time.time()        # 记录时间
print(a.device, t1 - t0, c.norm(2))     # c.norm(2)表示矩阵c的二范数
print(torch.cuda.is_available())
if torch.cuda.is_available():        # 当CUDA可用时返回True
    device = torch.device('cuda')       # 用GPU来运行
    a = a.to(device)
    b = b.to(device)

    # 初次调用GPU，需要数据传送，因此比较慢
    t0 = time.time()
    c = torch.matmul(a, b)
    t2 = time.time()
    print(a.device, t2 - t0, c.norm(2))

    # 这才是GPU处理数据的真实运行时间，当数据量越大，GPU的优势越明显
    t0 = time.time()
    c = torch.matmul(a, b)
    t2 = time.time()
    print(a.device, t2 - t0, c.norm(2))
