import math
import numpy as np
from model.dynamic_regularization import DynamicRegularization


def w_n(n, N, deta=0.4):
    temp_1 = deta * N / 2
    temp_2 = 1 / (math.sqrt(2 * math.pi) * temp_1)
    temp_3 = (-1 / 2) * math.pow((n - N / 2) / temp_1, 2)
    return temp_2 * math.exp(temp_3)


def get_deta_d(loss_list, deta_d, N=300):
    if len(loss_list) == 0 or len(loss_list) == 1:
        return 0

    w = np.array([w_n(n, N) for n in range(N + 1)])
    if len(loss_list) <= N + 1:
        # f_loss_1 = np.pad(np.array(loss_list), (0, N + 1 - len(loss_list)), 'constant', constant_values=1)
        # f_loss_2 = np.pad(np.array(loss_list[1:]), (0, N + 1 - len(loss_list[1:])), 'constant', constant_values=1)
        return deta_d if (loss_list[0] - loss_list[1]) < 0 else -deta_d
    else:
        f_loss_1 = np.array(loss_list[: N + 1])
        f_loss_2 = np.array(loss_list[1: N + 2])

        deta_loss1 = sum(f_loss_1 * w)
        deta_loss2 = sum(f_loss_2 * w)

        return deta_d if (deta_loss1 - deta_loss2) < 0 else -deta_d


def change_d(model, deta_d):
    if deta_d == 0:
        return model
    for layer in model.named_modules():
        if isinstance(layer[1], DynamicRegularization):
            layer[1].d = layer[1].d + deta_d
    return model
