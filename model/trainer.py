import tensorflow as tf
from model.loss_estimator import get_intra_loss
from model.cluster import commit_k_mean
from tensorflow.contrib.rnn import GRUCell
import numpy as np
from model.repres_learner import conv_act


def run(x, graph_filter, adj_sp, params, true_label=None, cluster_k=7):
    """
    initializing param
    feature and adj matrix run in graph convolution with ACT
    calculate loss after k-means in graph embedding
    loss function = λ_tig * L_tig + λ_sep * 1/L_sep

    :param x: X = {x_1, ..., x_M}, where M is the number of nodes, shape: M * D
    :param graph_filter: Normalized adjacency matrix used for graph convolution
    :param adj_sp: sparse adjacency of graph
    :param true_label: Ground truth label for each nodes
    :param cluster_k: nb. of classes
    :param params: all parameters used for model training
    :return: None
    """
    tf.set_random_seed(110)
    print(110)

    # ===== feed input =====
    # placeholders
    feature = tf.compat.v1.placeholder(tf.float32, shape=(None, x.shape[1]))
    plc_graph_filter = tf.compat.v1.placeholder(tf.float32, shape=graph_filter.shape)

    # for act
    # Accumulated halt signal that is below saturation value (variable threshold)
    p_t = tf.zeros(params["batch_size"], dtype=tf.float32, name="halting_probability")

    # Accumulated halt signal that is above saturation value (variable threshold)
    exceeded_p_t = tf.zeros_like(p_t, dtype=tf.float32, name="p_t_compare")
    # Index of halting convolution step
    n_t = tf.zeros(params["batch_size"], dtype=tf.float32, name="n_updates")

    # RNN model
    rnn_cell = GRUCell(params["hidden_size"])

    # Initialized state for rnn model
    state = rnn_cell.zero_state(params["batch_size"], tf.float32)

    # acculated output, i.e. y_t for {y_t^1, ..., y_t^N(t)}
    outputs_acc = tf.zeros_like(x, dtype=tf.float32, name="output_accumulator")

    # If a node have been already halted, the mask variable is assigned as '0', otherwise is '1'
    batch_mask = tf.fill([params["batch_size"]], True, name="batch_mask")

    # ===== learn representation =====
    embedding, _, pt_compare, pt, final_n_t, last_conv_embed \
        = conv_act(batch_mask, exceeded_p_t, p_t, n_t, plc_graph_filter,
                   state, feature, outputs_acc, params["max_conv_time"],
                   params["threshold"], rnn_cell)

    # get and record R_t
    final_r_t = 1 - pt

    # get intra_loss and inter_loss, drop other loss for save memory
    intra_loss, inter_loss, k_means_init_op, k_means_train_op = get_intra_loss(embedding, cluster_k)
    inter_loss_1 = params["inter_dist_coefficient"] * 1 / inter_loss
    loss = intra_loss + inter_loss_1

    # ===== backpropagation =====
    var_lr = tf.Variable(0.0, trainable=False)

    if params["max_grad_norm"] <= 0:
        optimizer = tf.compat.v1.train.AdamOptimizer(var_lr)
        train_step = optimizer.minimize(loss)
    else:
        tvars = tf.compat.v1.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), params["max_grad_norm"])
        optimizer = tf.compat.v1.train.AdamOptimizer(var_lr)
        train_step = optimizer.apply_gradients(zip(grads, tvars))

    # ===== Run already built computational graph =====
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    config = tf.compat.v1.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)

    with tf.compat.v1.Session(config=config) as sess:
        init_op = tf.compat.v1.global_variables_initializer()
        sess.run(init_op)

        # init k-means for get intra_loss & inter_loss
        sess.run(k_means_init_op, feed_dict={feature: x, plc_graph_filter: graph_filter})

        loss_list = []
        commit_msg_list = []

        print('----train start----')
        if params['auto_stop_flag']:
            print('----AUTO STOP----')

        for step in range(params["epoches"]):
            lr_decay = params["lr_decay"] ** max(step - params["max_epoch"], 0.0)
            sess.run(tf.compat.v1.assign(var_lr, params["lr"] * lr_decay))

            # start train k-means for get intra_loss & inter_loss
            _, total_loss, _, i_loss, in_loss, nt, embedding_y, rt \
                = sess.run([train_step, loss, k_means_train_op, intra_loss,
                            inter_loss_1, final_n_t, embedding, final_r_t],
                           feed_dict={feature: x, plc_graph_filter: graph_filter})

            train_msg = '[%d step(s), loss: %g, lr: %g] ' % (step + 1, total_loss, sess.run(var_lr))
            pt_msg = ' |intra_loss=' + str(i_loss) + ' |inter_loss=' + str(in_loss) + ' -> '

            # show number message of different convolution layers
            nt_msg = ' |Nt:' + str(show_nt_msg(nt))

            # Execute clustering and evaluation
            commit_msg = commit_k_mean(embedding_y, x, adj_sp, true_label, cluster_k, params)

            msg = train_msg + pt_msg + commit_msg + nt_msg

            print(msg)

            # auto stop
            if params['auto_stop_flag']:
                loss_list.append(total_loss)
                commit_msg_list.append(commit_msg)

                if auto_stop(loss_list, params):
                    print('select score '+str(step-params['stop_times']+2)+
                          ' step:\t'+commit_msg_list[-params['stop_times']])
                    break


def show_nt_msg(nt):
    """
    show number message of different halting convolution layers in nt

    :param nt: Index of halting convolution step
    :return nt_dict: number message of different halting convolution layers
    """

    nt_set = set(nt)
    nt_dict = {}

    for nt_node in nt_set:
        nt_dict[nt_node] = len(np.where(nt == nt_node)[0])

    return nt_dict


def auto_stop(loss_list, params):
    """

    collect count_loss_num step in loss list, and get std from them
    if their std continuely less than 'stop_mark' in 'stop_times',
    model will auto stop

    :param loss_list: collect loss
    :param params: all param msg

    :return auto stop flag (bool)
    """

    if len(loss_list) > params['count_loss_num']:
        array_loss = np.array(loss_list)
        # get newest losses
        array_loss = array_loss[-1*params['count_loss_num']:]
        loss_std = np.std(array_loss)

        # print('stop_mark=', loss_std, 'num=', params['stop_num'])

        # continuely
        # if loss_std occasionally less then stop_mark, and then it rises,
        # so set stop_num = 0
        if params['stop_num'] > 0:
            if loss_std > params['stop_mark']:
                params['stop_num'] = 0

        if loss_std < params['stop_mark']:
            params['stop_num'] += 1

            # if keep 5 times continuely meet the requirements
            if params['stop_num'] == params['stop_times']:
                return True
    return False
