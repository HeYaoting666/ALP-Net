import math
import torch.nn as nn
import torch
from model.dynamic_regularization import DynamicRegularization


class GFE_Block(nn.Module):
    def __init__(self, out_ch, R, d, A):
        super(GFE_Block, self).__init__()

        self.branch = self._make_branch(out_ch)
        self.f1_norm = nn.AdaptiveAvgPool2d(1)
        self.f_ex = nn.Sequential(
            nn.Conv2d(out_ch, out_ch // 4, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch // 4, out_ch, kernel_size=1, stride=1),
            nn.Sigmoid(),
        )
        self.dy = DynamicRegularization(R, d, A)

    @staticmethod
    def _make_branch(out_ch):
        return nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

    def p_normSE(self, y):
        k = self.f1_norm(y)  # [bs, c, 1, 1]
        alpha = self.f_ex(k)
        alpha_ = alpha.repeat(1, 1, list(y.shape)[2], list(y.shape)[3])

        return alpha_ * y

    def forward(self, x):
        residual = x
        x = self.branch(x)
        x_ = self.p_normSE(x)
        y = self.dy(x_)

        return residual + y


class MP(nn.Module):
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)


class Transition_Block(nn.Module):
    def __init__(self, c):
        super(Transition_Block, self).__init__()
        self.cv1 = nn.Conv2d(c, c, kernel_size=3, stride=2, padding=1)
        self.cv2 = nn.Conv2d(2 * c, 2 * c, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(2 * c)
        self.gelu = nn.GELU()

        self.mp = MP()

    def forward(self, x):
        # [32, 128, 64]  => [16, 64, 64]
        x_1 = self.mp(x)

        # [32, 128, 64]  => [16, 64, 64]
        x_2 = self.cv1(x)

        # [16, 64, 64] cat [16, 64, 64] => [16, 64, 128]
        x_3 = torch.cat([x_2, x_1], 1)
        x_3 = self.cv2(x_3)
        x_3 = self.bn(x_3)
        x_3 = self.gelu(x_3)
        return x_3


class ResBlock(nn.Module):
    def __init__(self, out_ch):
        super(ResBlock, self).__init__()

        self.branch = self._make_branch(out_ch)

    @staticmethod
    def _make_branch(out_ch):
        return nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

    def forward(self, x):
        residual = x
        x = self.branch(x)

        return x + residual


class SRCNet(nn.Module):
    def __init__(self, d, A, total_blocks, ch_list):
        super(SRCNet, self).__init__()
        self.d = d
        self.A = A
        self.R_dy = [(i + 1) / 12 for i in range(12)]
        self.block_idx = 0

        self.s_conv_1 = nn.Sequential(  # [bn, 3, 64, 256] -> [bn, 16, 32, 128]
            nn.Conv2d(3, ch_list[0], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ch_list[0]),
            nn.ReLU(inplace=True)
        )
        self.s_stage_1 = self._makelayer([ch_list[0], ch_list[1]], total_blocks // 3, transition=False)  # [bn, 16, 32, 128] -> [bn, 64, 32, 128]
        self.s_stage_2 = self._makelayer([ch_list[1], ch_list[2]], total_blocks // 3)  # [bn, 64, 32, 128] -> [bn, 128, 16, 64]
        self.s_stage_3 = self._makelayer([ch_list[2], ch_list[3]], total_blocks // 3)  # [bn, 128, 16, 64] -> [bn, 256, 8, 32]

        self.s_conv_2 = nn.Sequential(  # [bn, 256, 8, 32] -> [bn, 256, 1, 32]
            nn.Conv2d(ch_list[3], ch_list[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_list[3]),
            nn.GELU(),
            nn.Conv2d(ch_list[3], ch_list[3], kernel_size=(8, 1), stride=1),
            nn.BatchNorm2d(ch_list[3]),
            nn.GELU(),
        )
        self.s_stage_4 = self._makelayer2(ch_list[3], 2)  # [bn, 256, 1, 32] -> [bn, 256, 1, 32]

        self.classifier_1 = nn.Conv2d(ch_list[3], 34, kernel_size=(1, 7), stride=(1, 4))  # [bn, 256, 1, 32] -> [bn, 34, 1, 7]
        self.classifier_2 = nn.Conv2d(ch_list[3], 25, kernel_size=(1, 7), stride=(1, 4))  # [bn, 256, 1, 32] -> [bn, 25, 1, 7]
        self.classifier_3 = nn.Conv2d(ch_list[3], 35, kernel_size=(1, 7), stride=(1, 4))  # [bn, 256, 1, 32] -> [bn, 35, 1, 7]

        # Initialize parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _makelayer(self, channels, blocks, transition=True):
        if transition:
            layers = [Transition_Block(channels[0])]
        else:
            layers = [nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(channels[1]),
                      nn.GELU()]
        for i in range(blocks):
            layers.append(GFE_Block(channels[1], self.R_dy[self.block_idx], self.d, self.A))
            self.block_idx += 1
        return nn.Sequential(*layers)

    @staticmethod
    def _makelayer2(channels, blocks):
        layers = []
        for i in range(blocks):
            layers.append(ResBlock(channels))
            layers.append(ResBlock(channels))
            layers.append(nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=(1, 3), stride=1, padding=(0, 1)),
                nn.BatchNorm2d(channels),
                nn.GELU()
            ))
        return nn.Sequential(*layers)

    def forward(self, x):
        # ===================== Fe-Stage ====================== #
        x = self.s_conv_1(x)
        x = self.s_stage_1(x)
        x = self.s_stage_2(x)
        x = self.s_stage_3(x)
        # ===================== Fe-Stage ======================= #

        # ===================== Loc-Stage ====================== #
        x = self.s_conv_2(x)
        F_h = self.s_stage_4(x)
        # ===================== Loc-Stage ====================== #

        # ===================== Pred-Stage ===================== #
        out_1 = self.classifier_1(F_h).squeeze(2)[:, :, 0]
        out_2 = self.classifier_2(F_h).squeeze(2)[:, :, 1]
        out_3 = self.classifier_3(F_h).squeeze(2)[:, :, 2:].permute(0, 2, 1)
        # ===================== Pred-Stage ===================== #

        return [out_1, out_2, out_3]


# if __name__ == "__main__":
#     from torchsummary import summary
#
#     num_GFE_Block = 12
#     channel_list = [16, 64, 128, 256]  # [16, 32, 64, 128], [64, 128, 256, 512]
#     model = SRCNet(d=0, A=5, total_blocks=num_GFE_Block, ch_list=channel_list).to('cuda')
#     summary(model, (3, 64, 256), device='cuda')
