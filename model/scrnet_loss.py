import math
import torch
import torch.nn as nn


class SCRNetLoss(nn.Module):
    def __init__(self):
        super(SCRNetLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, pred, target):
        lpr_ch1 = pred[0]  # [bn, 34]
        lpr_ch2 = pred[1]  # [bn, 25]
        lpr_ch3 = pred[2]  # [bn, 5, 35]

        loss_1 = self.loss_fn(lpr_ch1, target[:, 0])
        loss_2 = self.loss_fn(lpr_ch2, target[:, 1])
        loss_3 = 0
        for i in range(5):
            loss_3 += self.loss_fn(lpr_ch3[:, i], target[:, i + 2])

        # loss = 0.75 * loss_1 + 0.75 * loss_2 + 1.5 * loss_3
        loss = loss_1 + loss_2 + loss_3
        return loss


def lr_warm_cos(model, lr_init, lr_min, total_steps):
    opt = torch.optim.Adam(params=model.parameters(), lr=lr_init)
    # warm_up_iter = int(total_steps * 0.1)
    # T_max = total_steps
    # if warm_up_iter == 0:
    #     lambda_lr = lambda cur_iter: (lr_min + 0.5 * (lr_max - lr_min) * (
    #             1.0 + math.cos((cur_iter - warm_up_iter) / (T_max - warm_up_iter) * math.pi))) / 0.1
    # else:
    #     lambda_lr = lambda cur_iter: (cur_iter / warm_up_iter) * (lr_max * 10) if cur_iter < warm_up_iter \
    #         else (lr_min + 0.5 * (lr_max - lr_min) * (
    #             1.0 + math.cos((cur_iter - warm_up_iter) / (T_max - warm_up_iter) * math.pi))) / 0.1
    # sdu = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda_lr)
    sdu = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=lr_min, last_epoch=-1)

    return opt, sdu


