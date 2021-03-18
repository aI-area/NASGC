from model.trainer import run
from util.data_processor import data_load
import warnings
import os


def train_start():
    warnings.filterwarnings("ignore")
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'    # set gpu

    data_name = 'citeseer'                  # wiki or citeseer
    g, adj, feature, true_label, cluster_k = data_load(data_name)

    if data_name == 'wiki':
        param = {
            'lr': 0.0001,                   # learning rate
            'epoches': 500,                 # max epoch
            'max_conv_time': 40,            # convolution layer
            'threshold': 0.99,              # accumulated smoothness value can not exceeds the threshold
            'batch_size': feature.shape[0], # all node num
            'hidden_size': 200,             # RNN layer hidden size (GRU)
            'inter_dist_coefficient': 10,   # lambda_sep (loss coefficient)
            'intra_dist_coefficient': 1,    # lambda_tig (loss coefficient)
            'epsilon': 1.84,                # 0.35d for dist-modularity
            'lr_decay': 0.96,               # learning rate decay  0.96
            'max_epoch': 10,                # using learning decay after 10 step
            'max_grad_norm': 1,             # default setting
            'auto_stop_flag': False,         # early stop
            'count_loss_num': 5,            # last 5 epochs
            'stop_mark': 0.001,             # auto stop threshold
            'stop_num': 0,                  # count times of std less than stop_mark
            'stop_times': 3                 # std need to less than stop_mark continuely 3 times

        }
    else:
        # citeseer
        param = {
            'lr': 0.003,                    # learning rate
            'epoches': 500,                 # max epoch
            'max_conv_time': 40,            # convolution layer
            'threshold': 0.99,              # accumulated smoothness value can not exceeds the threshold
            'batch_size': feature.shape[0], # all node num
            'hidden_size': 200,             # RNN layer hidden size (GRU)
            'inter_dist_coefficient': 350,  # lambda_sep (loss coefficient)
            'intra_dist_coefficient': 1,    # lambda_tig (loss coefficient)
            'epsilon': 2.71,                # 0.35d for dist-modularity
            'lr_decay': 1,                  # no using learning rate decayw
            'max_epoch': 0,                 # default setting
            'max_grad_norm': 0,             # default setting
            'auto_stop_flag': False,         # early stop
            'count_loss_num': 5,            # last 5 epochs
            'stop_mark': 0.001,             # auto stop threshold
            'stop_num': 0,                  # count times of std less than stop_mark
            'stop_times': 3                 # std need to less than stop_mark continuely 3 times
        }

    # ==== train start ====
    run(feature, g.toarray(), adj, param, true_label, cluster_k)


if __name__ == "__main__":
    train_start()
