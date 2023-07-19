import torch
from net.alp_net import SRCNet

iterations = 3000   # 重复计算的轮次

model = SRCNet(d=0, A=1, total_blocks=12, ch_list=[16, 64, 128, 256]).cuda()
model.load_state_dict(torch.load('weight/alp_net.pth'))

random_input = torch.randn(1, 3, 64, 256).cuda()
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

# GPU预热
for _ in range(50):
    _ = model(random_input)

# 测速
times = torch.zeros(iterations)     # 存储每轮iteration的时间
model.eval()
with torch.no_grad():
    for iter in range(iterations):
        starter.record()
        _ = model(random_input)
        ender.record()
        # 同步GPU时间
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)  # 计算时间
        times[iter] = curr_time
        # print(curr_time)

mean_time = times.mean().item()
print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000/mean_time))
