import os
import argparse
import torch.backends.cudnn as cudnn
import torch
from model.dynamic_regularization import DynamicRegularization

from datasets import LPRDataLoader
from net.alp_net import SRCNet
from model.alpnet_loss import lr_warm_cos, SCRNetLoss
from torch.autograd import Variable
from tqdm import tqdm
from utils.update_loss import change_d, get_deta_d
from utils.utils import accuracy, Logger, datetime


def main(args):
    for arg in vars(args):
        print(format(arg, '<20'), format(str(getattr(args, arg)), '<'))

    train_loader = LPRDataLoader(args.train_path, [256, 64], args.batch_size, num_workers=args.num_workers)
    val_loader = LPRDataLoader(args.val_path, [256, 64], args.batch_size, shuffle=False, num_workers=args.num_workers)
    loss_list = []

    model = SRCNet(d=args.d_init, A=args.A, total_blocks=args.total_blocks, ch_list=args.channel_list).cuda()
    if args.pretrained_model:
        f = open(os.path.join(args.pretrained_model_dir, 'model_loss.txt'))
        print("loss pretrained model successful!")
        for line in f:
            loss_list.append(float(line.strip()))
        model.load_state_dict(torch.load(args.pretrained_model))
        print("load pretrained model successful!")
    cudnn.benchmark = args.cudnn

    total_steps = len(train_loader) * args.epochs
    optimizer, scheduler = lr_warm_cos(model,
                                       args.lr_init,
                                       lr_min=args.lr_min,
                                       total_steps=total_steps,
                                       )
    loss_func = SCRNetLoss()
    headers = ["Epoch", "LearningRate", "TrainLoss", "TestLoss", "TrainAcc.", "TestAcc."]
    logger = Logger(args.checkpoint, headers)

    for epoch in range(args.epochs):
        model.train()
        train_loss, train_acc, train_n = 0, 0, 0
        bar = tqdm(total=len(train_loader), leave=False)

        for X, t in train_loader:
            X, t = Variable(X.cuda()), Variable(t.cuda())
            model = change_d(model, get_deta_d(loss_list, args.deta_d))  # 根据loss动态修改网络中s参数
            y = model(X)
            loss = loss_func(y, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_list.insert(0, loss.item())
            acc, _, _, _ = accuracy(y, t)

            train_acc += acc
            train_loss += loss.item()
            train_n += t.size(0)
            bar.set_description("Loss: {:.4f}, Accuracy: {:.2f}, lr: {:.6f}".format(
                train_loss / train_n, train_acc / train_n * 100, optimizer.param_groups[0]["lr"]), refresh=True)
            bar.update()
        bar.close()

        model.eval()
        val_loss, val_acc, val_n = 0, 0, 0
        for X, t in val_loader:
            with torch.no_grad():
                X, t = Variable(X.cuda()), Variable(t.cuda())
                y = model(X)
                loss = loss_func(y, t)
                acc, _, _, _ = accuracy(y, t)

                val_loss += loss.item()
                val_acc += acc
                val_n += t.size(0)

        if (epoch + 1) % args.snapshot_interval == 0:
            now_time = datetime.now().strftime("%m_%d-%H_%M_%S")
            save_path = os.path.join(args.checkpoint, now_time)
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_path, "{}.pth".format(epoch + 1)))

            f_s = open(os.path.join(save_path, "model_d.txt"), "w")
            for layer in model.named_modules():
                if isinstance(layer[1], DynamicRegularization):
                    f_s.write("{}".format(layer[1].d) + "\n")
            f_loss = open(os.path.join(save_path, "model_loss.txt"), "w")
            for llist in loss_list:
                f_loss.write("{}".format(llist) + '\n')

        lr = optimizer.param_groups[0]["lr"]
        logger.write(epoch + 1, lr, train_loss / train_n, val_loss / val_n,
                     train_acc / train_n * 100, val_acc / val_n * 100)


if __name__ == '__main__':
    # The model is trained for 80 epochs with a batch size of 256.
    # For the first 60 epochs, the hyperparameter A is set to 5,
    #                          the initial value of d_init is set to 0,
    #                          the deta_d is set to 0.
    # After it, load the first 60 epochs trained model, set d_init to 0.5 and deta_d to 3e-4.
    #
    # In the first 60 epochs, the initial learning rate is set to 1e−3, and then the learning rate will decay to 1e−7.
    # In the last 20 epochs, set the learning rate to 1e−5 and then decay to 1e−7.
    parser = argparse.ArgumentParser()
    parser.add_argument("--cudnn", type=bool, default=True)
    parser.add_argument("--checkpoint", type=str, default="./checkpoint")
    parser.add_argument("--snapshot_interval", type=int, default=5)
    # For Networks
    parser.add_argument("--total_blocks", type=int, default=12, help='total layers of GFE_Block')
    parser.add_argument("--channel_list", default=[16, 64, 128, 256], help='channel list of GFE_Block')
    parser.add_argument("--d_init", type=float, default=0)  # 0.5
    parser.add_argument("--A", type=float, default=5)
    parser.add_argument("--deta_d", type=float, default=0)  # 3e-4
    # For Training
    parser.add_argument("--train_path", default='E:/DataSets/MyDataSets/ccpd_lpr/train', help='the path of test data')
    parser.add_argument("--val_path", default='E:/DataSets/MyDataSets/ccpd_lpr/val', help='the path of test data')
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr_init", type=float, default=1e-3)
    parser.add_argument("--lr_min", type=float, default=1e-7)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument('--pretrained_model', default='', help='pretrained model path')  # 'checkpoint/05_06-21_02_30/60.pth'
    parser.add_argument('--pretrained_model_dir', default='', help='pretrained base model dir_path')  # 'checkpoint/05_06-21_02_30'
    args = parser.parse_args()
    main(args)
