#!/usr/bin/env python3
"""
@author: Juhyeon Lee [jh0104lee@gmail.com]
        Brain Signal Processing Lab [https://bspl-ku.github.io/]
        Department of Brain and Cognitive Engineering, Korea University, Seoul, Republic of Korea
"""

import numpy as np
import tensorflow as tf
import scipy.io as sio
import os

def init_data_params(directory_open='./data', subject_list=np.arange(10)+1, window_length_min=15/60, n_hid_units=[2000,2000], exclude_parcels=np.array([110,120,290,292,300],dtype=np.int16())-1):
    """
    Initialize parameters related to the input data and model architecture.

    :param directory_open: string, default='./data'
                            Directory including tvFC data.
    :param subject_list: array-like of shape (n_subjects,), default=[1,2,3,4,5,6,7,8,9,10]
                        Subject IDs for individual identification. We provide data from S1~S10.
    :param window_length_min: {15/60, 30/60, 1, 3, 5}, default=15/60
                                The length of sliding window in minute.
    :param n_hid_units: ndarray of shape (n_hid_layers,), default=[2000,2000]
                        The number of units in each hidden layer.
    :param exclude_parcels: ndarray of shape (n_exclude_parcels,), default=np.array([110,120,290,292,300]
                            Which parcels are excluded in calculating FC.
                            The default array indicates parcels not included in any of Yeo's 7 FNs.

    :return: directory_open, subject_list, n_subjects, window_length_min, n_samples_run, n_parcels, exclude_parcels, n_units, selected_units, n_hid_layers, n_sp_layers
    """
    # Total number of individuals to identify
    n_subjects = np.size(subject_list)

    TR = 0.72
    n_vols = 1200
    # Window length as number of volumes
    window_length = round(60*window_length_min/TR)
    # Non-overlapping sliding window
    step = window_length
    # Number of samples per run of each subject
    n_samples_run = int((n_vols-window_length)/step)+1

    # Glasser's MMP 360 parcels
    n_parcels = 360
    ids_mat = np.triu(np.ones((n_parcels,n_parcels),dtype=int),k=1)
    ids_mat[ids_mat==1] = np.arange(n_parcels*(n_parcels-1)/2,dtype=int)

    # Exclude parcels not needed
    select_mat = np.ones((n_parcels,n_parcels),dtype=int)
    for parcel in exclude_parcels:
        select_mat[parcel,:] = 0
        select_mat[:,parcel] = 0
    select_mat = np.triu(select_mat,k=1)
    # Selected input units after excluding unnecessary parcels
    selected_units = ids_mat[select_mat==1]
    len_connectivity = np.size(selected_units)

    # Number of units at each layer
    n_units = [[len_connectivity], n_hid_units, [n_subjects]]
    n_units = [item for sublist in n_units for item in sublist]

    # Number of hidden layers
    n_hid_layers = len(n_units)-1
    # Number of layers to control sparsity
    n_sp_layers = n_hid_layers-1

    return directory_open, subject_list, n_subjects, window_length_min, n_samples_run, n_parcels, exclude_parcels, n_units, selected_units, n_hid_layers, n_sp_layers



def init_train_params(directory_save='./results', n_epochs=2000, epoch_step_show=None, batch_size=10, lr_init=0.03, lr_min=0.0, lr_beginanneal=0, lr_drate=0.0, tg_sp_set=[0.8, 0.95], beta_max=[0.00015, 0.0005], beta_lr=None, gamma=1e-8, layer_activation=tf.nn.tanh, optimizer_algorithm='Momentum', momentum=0.9):
    """
    Initialize parameters related to the training and make result directory.

    :param directory_save: string, default='./results'
                            Root directory to save results.
    :param n_epochs: int, default=2000
                    How many epochs to train the DNN.
    :param epoch_step_show: int, default=None
                            How frequently print result on the screen.
                            None means printing 10 times in total regardless of how many epochs.
    :param batch_size: int, default=10
                        Mini-batch size, i.e, how many samples to train simultaneously in a training step.
    :param lr_init: float, default=0.03
                    Initial learning rate.
    :param lr_min: float, default=0.0
                    Minimum learning rate needed in case of learning rate annealing.
    :param lr_beginanneal: int, default=0
                            The epoch to start learning rate annealing.
                            0 means no learning rate annealing.
    :param lr_drate: float, default=0.0
                    Decay rate of learning rate in case of learning rate annealing.
                    Applied in forms of "lr = max(lr_min, (-lr_drate*(epoch+1) +(1+lr_drate*lr_beginanneal))*lr )".
    :param tg_sp_set: array-like of shape (n_hid_layers,), default=[0.8, 0.95]
                    Target Hoyer's sparsity level at each layer (i.e., [input, first hidden, second hidden, ...]) in the range [0, 1] (0:dense~1:sparse).
    :param beta_max: array-like of shape (n_hid_layers,), default=[0.00015, 0.0005]
                    Upper limit of beta value at each layer. L1 regulaization parameter.
    :param beta_lr: array-like of shape (n_hid_layers,), default=None
                    Learing rate (=incearing step) of beta value at each layer.
                    None means ten steps to reach to the maximum beta from zero.
    :param gamma: float, default=1e-8
                    L2 regulaization parameter.
    :param optimizer_algorithm: {'GradientDescent', 'Adagrad', 'Adam', 'Momentum', 'RMSProp'}, default='Momentum'
                                Optimizer algorithm.
    :param momentum: float, default=0.9
                    Momentum value in the range [0, 1].
                    Used for 'Momentum' optimizer.

    :return: final_directory, n_epochs, epoch_step_show, batch_size, lr_init, lr_min, lr_beginanneal, lr_drate, tg_sp_set, beta_max, beta_lr, gamma, layer_activation, optimizer_algorithm, momentum
    """

    import datetime
    import pytz

    # Step of epoch to print a result
    if epoch_step_show==None:
        epoch_step_show = n_epochs//10

    # How much allowing an increase of beta_vec value at each training iteration
    if beta_lr==None:
        beta_lr = [beta_max_l/10 for beta_max_l in beta_max]

    # Current time information and target sparsity will be used to create result directory
    dtime = datetime.datetime.now(tz=pytz.timezone('Asia/Seoul')) # or datetime.datetime.now(datetime.datetime.timezone(datetime.timedelta(hours=9)))
    selectedsp = ['%.2f-'%tg_sp for tg_sp in tg_sp_set]
    selectedsp[-1] = selectedsp[-1][:-1]
    selectedsp = ''.join(selectedsp)

    # Directory to save all results
    final_directory = os.path.join(directory_save, r'results_%04d%02d%02d_%02d%02d%02d_tgHSP_%s'%(dtime.year,dtime.month,dtime.day,dtime.hour,dtime.minute,dtime.second,selectedsp))
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)

    print()
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@@@@@@@@@         Starting         @@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('@@@@@@@@@@    '+'results_%04d%02d%02d_%02d%02d%02d_tgHSP_%s'%(dtime.year,dtime.month,dtime.day,dtime.hour,dtime.minute,dtime.second,selectedsp)+'    @@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')


    return final_directory, n_epochs, epoch_step_show, batch_size, lr_init, lr_min, lr_beginanneal, lr_drate, tg_sp_set, beta_max, beta_lr, gamma, layer_activation, optimizer_algorithm, momentum




def write_info(final_directory, n_subjects, window_length_min, n_samples_run, n_units, n_epochs, batch_size, lr_init, lr_min, lr_beginanneal, lr_drate, beta_max, beta_lr, optimizer_algorithm):
    """
    Write a text file about the overall information.
    """
    f = open(final_directory+'/init_info.txt','w')
    f.write('Number of subjects: '+str(n_subjects)+'\n')
    if isinstance(window_length_min,int):
        f.write('Window length (minute): '+str(window_length_min)+'\n')
    else:
        f.write('Window length (second): '+str(60*window_length_min)+'\n')
    f.write('Number of samples of each subject per run : '+str(n_samples_run)+'\n\n')
    f.write('Number of units: ' +str(n_units) + '\n')
    f.write('Number of epochs: '+str(n_epochs)+'\n')
    f.write('Mini-batch size: '+str(batch_size)+'\n')
    f.write('Initial learning rate: '+str(lr_init)+'\n')
    f.write('Begin annealing of learning rate: '+str(lr_beginanneal)+'\n')
    f.write('Decay rate of learning rate: '+str(lr_drate)+'\n')
    f.write('Minimum learning rate: '+str(lr_min)+'\n')
    f.write('Max beta: '+str(beta_max)+'\n')
    f.write('Learning rate of beta: '+str(beta_lr)+'\n')
    f.write('Optimizer algorithm: '+str(optimizer_algorithm)+'\n')
    f.close()



def load_data(directory_open='./data', subject_list=np.arange(10)+1, window_length_min=15/60, n_samples_run=57, selected_units=np.arange(62835)):
    """
    Load input data of DNN.

    :param directory_open: string, default='./data'
                            Directory including tvFC data.
    :param subject_list: array-like of shape (n_subjects,), default=[1,2,3,4,5,6,7,8,9,10]
                        Subject IDs for individual identification. We provide data from S1~S10.
    :param window_length_min: {15/60, 30/60, 1, 3, 5}, default=15/60
                                The length of sliding window in minute.
    :param n_samples_run: int, default=57
                            Number of samples per run of each subject.
    :param selected_units:, ndarry of shape (n_input_units,), default=np.arange(62835)
                            Selected input units after excluding unnecessary parcels.

    :return: x_train, y_train, x_valid, y_valid, x_test, y_test
    """

    from scipy import stats

    # Total number of individuals to identify
    n_subjects = len(subject_list)

    # Training set: RS1_RL+RS1_LR
    x_train = np.zeros((2*n_subjects*n_samples_run, len(selected_units)))
    y_train = np.zeros((2*n_subjects*n_samples_run, n_subjects))
    # Validation set: RS2_RL
    x_valid = np.zeros((n_subjects*n_samples_run, len(selected_units)))
    y_valid = np.zeros((n_subjects*n_samples_run, n_subjects))
    # Test set: RS2_LR
    x_test = np.zeros((n_subjects*n_samples_run, len(selected_units)))
    y_test = np.zeros((n_subjects*n_samples_run, n_subjects))

    for i_sbj,sbj in enumerate(subject_list):
        for i_run,enc_phase_run in enumerate(['RL','LR']):
            # Load the first session data (RS1) for training set
            sess1_tmp = directory_open+'/RS1/'+enc_phase_run+'/S'+str(sbj).zfill(3)+'_dFC_r.mat'
            sess1_tmp = sio.loadmat(sess1_tmp)
            sess1_tmp = np.array(sess1_tmp['dfc'])
            # Stack after loading a run since training set includes two runs of a session
            stack_loc = i_run*n_subjects*n_samples_run
            x_train[(stack_loc+i_sbj*n_samples_run):(stack_loc+(i_sbj+1)*n_samples_run),:] = sess1_tmp[:,selected_units]
            # One-hot
            y_train[(stack_loc+i_sbj*n_samples_run):(stack_loc+(i_sbj+1)*n_samples_run),i_sbj] = 1


            # Load the second session data (RS2) for validation/test sets
            sess2_tmp = directory_open+'/RS2/'+enc_phase_run+'/S'+str(sbj).zfill(3)+'_dFC_r.mat'
            sess2_tmp = sio.loadmat(sess2_tmp)
            sess2_tmp = np.array(sess2_tmp['dfc'])
            if enc_phase_run=='RL':
                x_valid[(i_sbj*n_samples_run):((i_sbj+1)*n_samples_run),:] = sess2_tmp[:,selected_units]
                # One-hot
                y_valid[(i_sbj*n_samples_run):((i_sbj+1)*n_samples_run),i_sbj] = 1
            elif enc_phase_run=='LR':
                x_test[(i_sbj*n_samples_run):((i_sbj+1)*n_samples_run),:] = sess2_tmp[:,selected_units]
                # One-hot
                y_test[(i_sbj*n_samples_run):((i_sbj+1)*n_samples_run),i_sbj] = 1

        if (i_sbj+1)%1==0:
           print("subject %d loaded"%sbj)

    # z-scoring across features within a sample
    x_train = np.arctanh(x_train)
    x_train = stats.zscore(x_train, axis=1)
    x_valid = np.arctanh(x_valid)
    x_valid = stats.zscore(x_valid, axis=1)
    x_test = np.arctanh(x_test)
    x_test = stats.zscore(x_test, axis=1)

    # print(np.shape(x_train))
    # print(np.shape(y_train))
    # print(np.shape(x_valid))
    # print(np.shape(y_valid))
    # print(np.shape(x_test))
    # print(np.shape(y_test))

    return x_train, y_train, x_valid, y_valid, x_test, y_test


def init_pretrained_dnn(n_units, n_hid_layers, layer_activation=tf.nn.tanh):
    """
    Build a DNN with same structure as pretrained model (to restore pretrained Tensorflow variables easily).

    :param n_units: array-like of shape (n_layers,)
                    Number of units at each layer.
    :param n_hid_layers: int
                        Number of hidden layers
    :param layer_activation: DNN operation in Tensorflow form, default=tf.nn.tanh
                            Activation function applied between layers.

    :return: X, Y, dnn, dnn_bn, dnn_bn_act, unit_vec_split_pre, n_units_pre, is_train
    """

    from functools import partial

    # Clear the default graph stack and resets the global default graph
    tf.reset_default_graph()

    # Index list used to split a vectorized placeholder later
    n_units_pre = n_units[:-1]+[300]
    unit_vec_split_pre = [int(np.sum(n_units_pre[1:(layer+1)])) for layer in range(n_hid_layers+1)]

    # Training flag
    is_train = tf.placeholder(tf.bool)

    # Placeholder to take input data
    X = tf.placeholder(tf.float32, [None,n_units_pre[0]])
    # Placeholder to take labels
    Y = tf.placeholder(tf.float32, [None,n_units_pre[-1]])

    # DNN layer by layer
    dnn = [0.0]*n_hid_layers
    dnn_bn = [0.0]*n_hid_layers
    dnn_bn_act = [0.0]*n_hid_layers

    # Batch normalization function
    batch_norm_layer = partial(tf.contrib.layers.batch_norm, center=True, scale=True, is_training=is_train)

    for layer in range(n_hid_layers-1):
        # Input layer
        if layer==0:
            dnn[layer] = tf.layers.dense(
                                        X,
                                        n_units_pre[layer+1],
                                        activation=None,
                                        use_bias=True,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
                                        bias_initializer=tf.zeros_initializer(),
                                        kernel_regularizer=None,
                                        bias_regularizer=None,
                                        activity_regularizer=None,
                                        trainable=True,
                                        name="hidden%d_layer"%(layer+1),
                                        reuse=False)

        # Hidden layers
        else:
            dnn[layer] = tf.layers.dense(
                                        dnn_bn_act[layer-1],
                                        n_units_pre[layer+1],
                                        activation=None,
                                        use_bias=True,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
                                        bias_initializer=tf.zeros_initializer(),
                                        kernel_regularizer=None,
                                        bias_regularizer=None,
                                        activity_regularizer=None,
                                        trainable=True,
                                        name="hidden%d_layer"%(layer+1),
                                        reuse=False)
        # Apply batch normalization
        dnn_bn[layer] = batch_norm_layer(dnn[layer])
        # Apply activation function
        dnn_bn_act[layer] = layer_activation(dnn_bn[layer])


    # Output layer
    dnn[-1] = tf.layers.dense(
                            dnn_bn_act[-2],
                            n_units_pre[-1],
                            activation=None,
                            use_bias=True,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
                            bias_initializer=tf.zeros_initializer(),
                            kernel_regularizer=None,
                            bias_regularizer=None,
                            activity_regularizer=None,
                            trainable=True,
                            name="hidden%d_layer"%n_hid_layers,
                            reuse=False)

    # Output layer does not apply batch normalization
    dnn_bn[-1] = dnn[-1]
    # No activation function either
    dnn_bn_act[-1] = dnn_bn[-1]

    return X, Y, dnn, dnn_bn, dnn_bn_act, unit_vec_split_pre, n_units_pre, is_train


def init_dnn_last_layer(n_units, n_hid_layers, dnn, dnn_bn, dnn_bn_act):
    """
    Substitute the last layer and its related variables with a new one.

    :param n_units: array-like of shape (n_layers,)
                    Number of units at each layer.
    :param n_hid_layers: int
                        Number of hidden layers
    :param dnn: tf.Tensor
                A built DNN model.
    :param dnn_bn: tf.Tensor
                Batch normalization applied DNN.
    :param dnn_bn_act: tf.Tensor
                Activation function applied DNN.

    :return: Y_new, dnn_new, dnn_bn_new, dnn_bn_act_new, unit_vec_split
    """

    # Index list used to split a vectorized placeholder later
    unit_vec_split = [int(np.sum(n_units[1:(layer+1)])) for layer in range(n_hid_layers+1)]

    # Placeholder to take labels
    Y_new = tf.placeholder(tf.float32, [None,n_units[-1]],name='Y_new')

    # Copy initial model
    dnn_new = dnn[:]
    dnn_bn_new = dnn_bn[:]
    dnn_bn_act_new = dnn_bn_act[:]

    # Re-create output layer
    dnn_new[-1] = tf.layers.dense(
                            dnn_bn_act_new[-2],
                            n_units[-1],
                            activation=None,
                            use_bias=True,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
                            bias_initializer=tf.zeros_initializer(),
                            kernel_regularizer=None,
                            bias_regularizer=None,
                            activity_regularizer=None,
                            trainable=True,
                            name="hidden%d_layer_new"%n_hid_layers,
                            reuse=False)

    # Output layer does not apply batch normalization
    dnn_bn_new[-1] = dnn_new[-1]
    # No activation function either
    dnn_bn_act_new[-1] = dnn_bn_new[-1]

    return Y_new, dnn_new, dnn_bn_new, dnn_bn_act_new, unit_vec_split


def init_regularizations(n_units, unit_vec_split, n_hid_layers, n_sp_layers, dnn):
    """
    Add regularization terms to cost.

    :param n_units: array-like of shape (n_layers,)
                    Number of units at each layer.
    :param unit_vec_split: array-like of shape (n_layers,)
                            Index list used to split a vectorized placeholder later.
                            For example, given n_units=[62835, 2000, 2000, 10], unit_vec_split=[0, 2000, 4000, 4010].
    :param n_hid_layers: int
                        Number of hidden layers
    :param n_sp_layers: int
                        Number of layers to control sparsity
    :param dnn: tf.Tensor
                A built DNN model.

    :return: Beta, L1_loss_total, Gamma, L2_loss_total
    """

    # Weight parameters
    w = [tf.get_default_graph().get_tensor_by_name(os.path.split(dnn[layer].name)[0]+'/kernel:0') for layer in range(n_hid_layers)]

    # L1 loss term by multiplying betas (vector values as many as units) and L1 norm of weight for each layer
    Beta = tf.placeholder(tf.float32, [np.sum(n_units[1:(n_sp_layers+1)])])
    L1_loss = [tf.reduce_sum(tf.matmul(tf.abs(w[layer]), tf.reshape(Beta[unit_vec_split[layer]:unit_vec_split[layer+1]],[-1,1]))) for layer in range(n_sp_layers)]
    L1_loss_total = tf.reduce_sum(L1_loss)

    # L2 loss term by multiplying gamma and L2 norm
    Gamma = tf.placeholder(tf.float32)
    L2_loss = [tf.reduce_sum(tf.square(w[layer])) for layer in range(n_hid_layers)]
    L2_loss_total = Gamma*tf.reduce_sum(L2_loss)

    return Beta, L1_loss_total, Gamma, L2_loss_total


def init_cost_optimizer(n_units, unit_vec_split, n_hid_layers, n_sp_layers, dnn, Y, optimizer_algorithm='Momentum', momentum=0.9):
    """
    Initialize operations and variables related to training.

    :param n_units: array-like of shape (n_layers,)
                    Number of units at each layer.
    :param unit_vec_split: array-like of shape (n_layers,)
                            Index list used to split a vectorized placeholder later.
                            For example, given n_units=[62835, 2000, 2000, 10], unit_vec_split=[0, 2000, 4000, 4010].
    :param n_hid_layers: int
                        Number of hidden layers
    :param n_sp_layers: int
                        Number of layers to control sparsity
    :param dnn: tf.Tensor
                A built DNN model.
    :param Y: tf.Tensor
                Placeholder to take labels.
    :param optimizer_algorithm: {'GradientDescent', 'Adagrad', 'Adam', 'Momentum', 'RMSProp'}, default='Momentum'
                                Optimizer algorithm.
    :param momentum: float, default=0.9
                    Momentum value in the range [0, 1].
                    Used for 'Momentum' optimizer.

    :return: lr, hsp, beta, beta_vec, plot_hsp, plot_beta, plot_lr, plot_tot_cost, plot_train_err, plot_val_err, plot_test_err

    """
    # Add regularization terms to cost
    Beta, L1_loss_total, Gamma, L2_loss_total = init_regularizations(n_units, unit_vec_split, n_hid_layers, n_sp_layers, dnn)

    # Softmax adds up the evidence of our input being in certain classes, and converts that evidence into probabilities
    cost_org = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=dnn[-1], labels=Y))
    cost = cost_org + L1_loss_total + L2_loss_total

    # To minimize the effect of different mini-batch size
    N_batches = tf.placeholder(tf.float32)
    cost = cost/N_batches

    # For the batch normalization update
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):

        # Placeholder to update learning rate (Learning rate decaying)
        Lr = tf.placeholder(tf.float32)

        with tf.name_scope('train'):
            # Select the optimizer
            if optimizer_algorithm=='GradientDescent':
                optimizer = tf.train.GradientDescentOptimizer(Lr).minimize(cost)
            elif optimizer_algorithm=='Adagrad':
                optimizer = tf.train.AdagradOptimizer(Lr).minimize(cost)
            elif optimizer_algorithm=='Adam':
                optimizer = tf.train.AdamOptimizer(Lr).minimize(cost)
            elif optimizer_algorithm=='Momentum':
                optimizer = tf.train.MomentumOptimizer(Lr,momentum).minimize(cost)
            elif optimizer_algorithm=='RMSProp':
                optimizer = tf.train.RMSPropOptimizer(Lr).minimize(cost)

    return cost, optimizer, Lr, Beta, Gamma, N_batches



def init_other_variables(n_subjects, lr_init, n_units, n_sp_layers):

    """
    Create lists and arrays to store results obtained during training.

    :param n_subjects: int
                        Total number of individuals to identify.
    :param lr_init: float
                    Initial learning rate.
    :param n_units: array-like of shape (n_layers,)
                    Number of units at each layer.
    :param n_sp_layers: int
                        Number of layers to control sparsity

    :return: lr, hsp, beta, beta_vec, plot_hsp, plot_beta, plot_lr, plot_tot_cost, plot_train_err, plot_val_err, plot_test_err
    """

    # To update current sparsity and beta
    hsp = [np.zeros(n_units[layer+1], dtype=np.float32) for layer in range(n_sp_layers)]
    beta = [np.zeros(n_units[layer+1], dtype=np.float32) for layer in range(n_sp_layers)]
    beta_vec = np.zeros(np.sum(n_units[1:(n_sp_layers+1)]), dtype=np.float32)

    # To store and plot results
    plot_hsp = [np.zeros(n_units[layer+1], dtype=np.float32) for layer in range(n_sp_layers)]
    plot_beta = [np.zeros(n_units[layer+1], dtype=np.float32) for layer in range(n_sp_layers)]
    plot_lr = np.zeros((1), dtype=np.float32)
    plot_tot_cost = np.zeros((1), dtype=np.float32)
    plot_train_err = np.zeros((n_subjects,1), dtype=np.float32)
    plot_val_err = np.zeros((n_subjects,1), dtype=np.float32)
    plot_test_err = np.zeros((n_subjects,1), dtype=np.float32)

    # initialize learning rate
    lr = lr_init

    return lr, hsp, beta, beta_vec, plot_hsp, plot_beta, plot_lr, plot_tot_cost, plot_train_err, plot_val_err, plot_test_err



def Hoyers_sparsity_control(beta_lr_l, W_l, beta_l, beta_max_l, tg_hsp_l):
    """
    Calculate Hoyer's sparsity of weight parameters at a layer, and update beta.

    :param beta_lr_l: float
                        Learing rate (=incearing step) of beta value at the current layer.
    :param W_l: ndarray of shape (dim, n_units)
                Weight matrix at the current layer.
    :param beta_l: ndarray of shape (n_units,)
                    Beta values at the current layer.
    :param beta_max_l: float
                        Upper limit of beta value at the current layer.
    :param tg_hsp_l: float
                    Target Hoyer's sparsity level at the current layer in the range [0, 1] (0:dense~1:sparse).

    :return: [hsp_l, beta_l_update]
    """

    # Linear algebra module for calculating L1 and L2 norm
    from numpy import linalg as LA

    # Weight matrix in the shape of (dimension, number of units)
    [dim, n_units] = W_l.shape

    # L1 and L2 norm of weight matrix
    L1 = LA.norm(W_l, 1, axis=0)
    L2 = LA.norm(W_l, 2, axis=0)

    # Hoyer's sparsity level at the current layer
    hsp_l = np.zeros((1,n_units))
    hsp_l = (np.sqrt(dim)-(L1/L2)) /(np.sqrt(dim)-1)

    # Update beta based on the current sparsity level
    flag_beta_update = np.sign(hsp_l-np.ones(n_units)*tg_hsp_l)
    beta_l_update = beta_l -beta_lr_l*flag_beta_update

    # Trim beta to be in the range of (0.0, beta_max_l)
    beta_l_update[beta_l_update<0.0] = 0.0
    beta_l_update[beta_l_update>beta_max_l] = beta_max_l

    return [hsp_l, beta_l_update]



def get_err(sess, hsp, epoch, epoch_step_show, n_subjects, n_samples_run, x_train, y_train, x_valid, y_valid, x_test, y_test, plot_train_err, plot_val_err, plot_test_err,error, X, Y, is_train):
    """
    Get the classification error rate.
    """

    err_tr_out = np.zeros((n_subjects,1), dtype=np.float32)
    err_vd_out = np.zeros((n_subjects,1), dtype=np.float32)
    err_ts_out = np.zeros((n_subjects,1), dtype=np.float32)

    for sbj in range(n_subjects):

        # To load data subject by subject
        loc_sbj = sbj*n_samples_run +np.arange(n_samples_run)
        x_train_sbj = np.vstack([x_train[loc_sbj,:], x_train[np.size(x_train,0)//2+loc_sbj,:]])
        y_train_sbj = np.vstack([y_train[loc_sbj,:], y_train[np.size(x_train,0)//2+loc_sbj,:]])
        x_valid_sbj = x_valid[loc_sbj,:]
        y_valid_sbj = y_valid[loc_sbj,:]
        x_test_sbj = x_test[loc_sbj,:]
        y_test_sbj = y_test[loc_sbj,:]

        # Get error rate
        err_tr_out[sbj] = sess.run(error, {X:x_train_sbj, Y:y_train_sbj, is_train:False})
        err_vd_out[sbj] = sess.run(error, {X:x_valid_sbj, Y:y_valid_sbj, is_train:False})
        err_ts_out[sbj] = sess.run(error, {X:x_test_sbj, Y:y_test_sbj, is_train:False})

    # Merge all subjects' error rates
    plot_train_err = np.hstack([plot_train_err, err_tr_out])
    plot_val_err = np.hstack([plot_val_err, err_vd_out])
    plot_test_err = np.hstack([plot_test_err, err_ts_out])

    # Print average error rate
    if ((epoch+1)%epoch_step_show==0):
        print("<epoch {:03d}> tr err:".format(epoch+1),"{:.3f}".format(np.mean(err_tr_out)),
              "/vd err:","{:.3f}".format(np.mean(err_vd_out)),"/ts err:","{:.3f}".format(np.mean(err_ts_out)),
              "/HSP:",["{:.2f}".format(np.mean(sp)) for sp in hsp])

    return plot_train_err, plot_val_err, plot_test_err



def plot_save_results(sess, final_directory, tg_sp_set, plot_lr, plot_tot_cost, plot_train_err, plot_val_err, plot_test_err, plot_beta, plot_hsp, lr_init, beta_max, dnn, n_units, n_hid_layers, n_sp_layers):
    """
    Plot results and save mat files.
    """

    import matplotlib
    # matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Plot the change of learning rate
    plot = plt.figure()
    plt.title("Learning rate plot", fontsize=16)
    plt.ylim(0.0, lr_init*1.2)
    plt.plot(plot_lr)
    plt.grid()
    plt.savefig(final_directory+'/learning_rate.png')
    plt.show()
    plt.close(plot)

    # Plot the change of cost
    plot = plt.figure()
    plt.title("Cost plot (log scale)", fontsize=16)
    plt.plot(plot_tot_cost)
    plt.yscale('log')
    plt.grid()
    plt.savefig(final_directory+'/cost.png')
    plt.show()
    plt.close(plot)

    # Plot the change of beta
    plot = plt.figure()
    for layer in range(n_sp_layers):
        plt.plot(np.mean(plot_beta[layer],1), label='layer%d'%(layer+1))
        plt.hold(True) if int(matplotlib.__version__[0])<3 else None
    plt.title("Beta plot", fontsize=16)
    plt.ylim(0.0, np.max(beta_max)*1.2)
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(final_directory+'/beta.png')
    plt.show(block=False)
    plt.close(plot)

    # Plot the change of beta by unit
    plot = plt.figure()
    for layer in range(n_sp_layers):
        plt.plot(plot_beta[layer])
        plt.hold(True) if int(matplotlib.__version__[0])<3 else None
    plt.grid()
    plt.title("Beta plot (by unit)", fontsize=16)
    plt.ylim(0.0, np.max(beta_max)*1.2)
    plt.savefig(final_directory+'/beta_hid_unit.png')
    plt.show()
    plt.close(plot)


    # Plot the change of Hoyer's sparsity
    plot = plt.figure()
    for layer in range(n_sp_layers):
        plt.plot(np.mean(plot_hsp[layer],1), label='layer%d'%(layer+1))
        plt.hold(True) if int(matplotlib.__version__[0])<3 else None
    plt.grid()
    plt.title("Hoyer's sparsity plot", fontsize=16)
    plt.ylim(0.0, 1.0)
    plt.legend(loc='best')
    plt.savefig(final_directory+'/HSP_'+'_'.join(["{:.2f}".format(np.mean(sp[-1])) for sp in plot_hsp])+'.png')
    plt.show(block=False)
    plt.close(plot)

    # Plot the change of Hoyer's sparsity by unit
    plot = plt.figure()
    for layer in range(n_sp_layers):
        plt.plot(plot_hsp[layer])
        plt.hold(True) if int(matplotlib.__version__[0])<3 else None
    plt.grid()
    plt.title("Hoyer's sparsity plot (by unit)", fontsize=16)
    plt.ylim(0.0, 1.0)
    plt.savefig(final_directory+'/HSP_'+'_'.join(["{:.2f}".format(np.mean(sp[-1])) for sp in plot_hsp])+'_hid_unit.png')
    plt.show()
    plt.close(plot)


    # Plot the change of error rates
    plot = plt.figure()
    plt.title("Error rate plot", fontsize=16)
    plt.plot(np.mean(plot_train_err, axis=0), label='Training')
    plt.hold(True) if int(matplotlib.__version__[0])<3 else None
    plt.plot(np.mean(plot_val_err, axis=0), label='Validation')
    plt.hold(True) if int(matplotlib.__version__[0])<3 else None
    plt.plot(np.mean(plot_test_err, axis=0), label='Test')
    plt.grid()
    plt.ylim(0.0, 1.0)
    plt.legend(loc='upper left')
    plt.savefig(final_directory+'/error_rates_tr{:.3f}_vd{:.3f}_ts{:.3f}'.format(
                    np.mean(plot_train_err[:,-1],axis=0),np.mean(plot_val_err[:,-1],axis=0),np.mean(plot_test_err[:,-1],axis=0))+'.png')
    plt.show()
    plt.close(plot)

    # Save error rates as mat files
    sio.savemat(final_directory+"/result_training_err.mat", mdict={'trainErr': plot_train_err})
    sio.savemat(final_directory+"/result_validation_err.mat", mdict={'validationErr': plot_val_err})
    sio.savemat(final_directory+"/result_test_err.mat", mdict={'testErr': plot_test_err})

    # Save weight and bias as mat files
    w = [sess.run(tf.get_default_graph().get_tensor_by_name(os.path.split(dnn[layer].name)[0]+'/kernel:0')) for layer in range(n_hid_layers)]
    b = [sess.run(tf.get_default_graph().get_tensor_by_name(os.path.split(dnn[layer].name)[0]+'/bias:0')) for layer in range(n_hid_layers)]
    sio.savemat(final_directory+"/result_weight.mat", mdict={'weight':w})
    sio.savemat(final_directory+"/result_bias.mat", mdict={'bias':b})


