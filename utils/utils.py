import os
import torch
import json
from datetime import datetime
from config import *
import cv2


def get_dir_paths(path):
    dir_paths = []
    for root, dirs, files in os.walk(path):
        for name in dirs:
            dir_paths.append(os.path.join(root, name))
    return dir_paths


def collate_fn(batch):
    imgs = []
    labels = []
    for _, sample in enumerate(batch):
        img, label = sample
        imgs.append(torch.from_numpy(img))
        labels.append(label)

    return torch.stack(imgs, 0), torch.tensor(labels)


def get_lpr(ch1, ch2, ch3):
    ch_list = [CHARS_1[ch1], CHARS_2[ch2]]
    for c in ch3:
        ch_list.append(CHARS_3[c])

    return ch_list


def accuracy(pred, target, chinese_region=True):
    acc = 0
    province_error = 0
    ch_error = 0
    pro_ch_error = 0

    lpr_ch1 = torch.argmax(pred[0], dim=1, keepdim=True)  # [bn, 1]
    lpr_ch2 = torch.argmax(pred[1], dim=1, keepdim=True)  # [bn, 1]
    lpr_ch3 = torch.argmax(pred[2], dim=2)  # [bn, 5]

    lpr_ch = torch.cat((lpr_ch1, lpr_ch2), 1)
    lpr_ch = torch.cat((lpr_ch, lpr_ch3), 1)
    for i in range(target.size(0)):
        if chinese_region:
            if lpr_ch[i].eq(target[i]).sum().item() == 7:
                acc += 1
            else:
                if lpr_ch[i][0] != target[i][0] and lpr_ch[i][1:].eq(target[i][1:]).sum().item() == 6:
                    province_error += 1
                elif lpr_ch[i][0] == target[i][0] and lpr_ch[i][1:].eq(target[i][1:]).sum().item() != 6:
                    ch_error += 1
                else:
                    pro_ch_error += 1
        else:
            if lpr_ch[i][1:].eq(target[i][1:]).sum().item() == 6:
                acc += 1
            else:
                ch_error += 1

    return acc, province_error, ch_error, pro_ch_error


class Logger:
    def __init__(self, log_dir, headers):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.f = open(os.path.join(log_dir, "log.txt"), "w")
        header_str = "\t".join(headers + ["EndTime."])
        self.print_str = "\t".join(["{}"] + ["{:.6f}"] * (len(headers) - 1) + ["{}"])

        self.f.write(header_str + "\n")
        self.f.flush()
        print(header_str)

    def write(self, *args):
        now_time = datetime.now().strftime("%m/%d %H:%M:%S")
        self.f.write(self.print_str.format(*args, now_time) + "\n")
        self.f.flush()
        print(self.print_str.format(*args, now_time))

    def write_hp(self, hp):
        json.dump(hp, open(os.path.join(self.log_dir, "hp.json"), "w"))

    def close(self):
        self.f.close()
