import argparse
import os.path
import torch

from datasets import LPRDataLoader, get_dir_paths
from net.alp_net import AlpNet
from torch.autograd import Variable
from utils.utils import accuracy


def main(args):
    test_dirs = get_dir_paths(args.test_path)
    test_group = list(map(os.path.basename, test_dirs))
    test_loaders = [LPRDataLoader(path, [256, 64], args.batch_size, shuffle=False, num_workers=2) for path in test_dirs]

    model = AlpNet(d=args.d_init, A=args.A, total_blocks=args.total_blocks, ch_list=args.channel_list).cuda()
    if args.pretrained_model:
        model.load_state_dict(torch.load(args.pretrained_model))
        print("load pretrained model successful!")

    model.eval()
    total_acc, total_n, total_pro_chinese_error = 0, 0, 0
    for i, test_loader in enumerate(test_loaders):
        print("The {} group is:".format(test_group[i]))
        test_acc, test_n = 0, 0
        pro_error, ch_error, pro_ch_error = 0, 0, 0
        for X, t in test_loader:
            with torch.no_grad():
                X, t = Variable(X.cuda()), Variable(t.cuda())
                y = model(X)

                TP, TF1, TF2, TF3 = accuracy(y, t)
                test_acc += TP
                pro_error += TF1
                ch_error += TF2
                pro_ch_error += TF3
                test_n += t.size(0)
        error_n = pro_error + ch_error + pro_ch_error
        chinese_error = pro_error + pro_ch_error
        if error_n == 0:
            print("ACC: 100%")
        else:
            print("ACC: {:.2f}%, (PROVINCE_ERROR: {:.2f}%, CHAR_ERROR: {:.2f}%, BOTH_ERROR: {:.2f}%)(100%)".format(
                test_acc / test_n * 100,
                pro_error / error_n * 100,
                ch_error / error_n * 100,
                pro_ch_error / error_n * 100
            ))
        total_acc += test_acc
        total_n += test_n
        total_pro_chinese_error += chinese_error

    if total_n == total_acc:
        print("Total ACC: {:.2f}%, Ce: {:.2f}%".format(
            total_acc / total_n * 100,
            0.0
        ))
    else:
        print("Total ACC: {:.2f}%, Ce: {:.2f}%".format(
            total_acc / total_n * 100,
            total_pro_chinese_error / (total_n - total_acc) * 100
        ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # For Networks
    parser.add_argument("--total_blocks", type=int, default=12, help='total layers of GFE_Block')
    parser.add_argument("--channel_list", default=[16, 64, 128, 256], help='channel list of GFE_Block')
    parser.add_argument("--d_init", type=float, default=0)
    parser.add_argument("--A", type=float, default=5)
    # For Testing
    parser.add_argument("--test_path", default='E:/DataSets/MyDataSets/clpd/', help='the path of test data')
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument('--pretrained_model', default='./weight/alp_net.pth', help='pretrained base model')
    args = parser.parse_args()
    main(args)
    # test_path = 'E:/DataSets/MyDataSets/ccpd_lpr/test', 'E:/DataSets/MyDataSets/clpd/', 'E:/DataSets/MyDataSets/crpd_all/', './test_img/'
    # total_block = 6, 9, 12, 15
    # channel_list = [16, 32, 64, 128], [16, 64, 128, 256], [64, 128, 256, 512]
