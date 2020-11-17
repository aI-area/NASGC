from model.trainer import run
from util.data_processor import data_load
import warnings
import os


def train_start():
    warnings.filterwarnings("ignore")
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    g, adj, feature, true_label, cluster_k = data_load('cora')

    param = {
        'lr': 0.01,
        'epoches': 500,
        'max_conv_time': 40,            # convolution layer
        'threshold': 0.99,              # accumulated smoothness value can not exceeds the threshold
        'batch_size': feature.shape[0], # all node num
        'hidden_size': 200,             # RNN layer hidden size (GRU)
        'inter_dist_coefficient': 50,   # lambda_sep (loss coefficient)
        'intra_dist_coefficient': 1,    # lambda_tig (loss coefficient)
        'auto_stop_flag': True,         # early stop
        'count_loss_num': 5,            # last 5 epochs
        'stop_mark': 0.001,             # auto stop threshold
        'stop_num': 0,                  # count times of std less than stop_mark
        'stop_times': 3                 # std need to less than stop_mark continuely 3 times
    }

    # ==== train start ====
    run(feature, g.toarray(), param, true_label, cluster_k)


if __name__ == "__main__":
    train_start()
