from dataset import *
import os, sys, getopt
from conv_net_classes import *
from collections import OrderedDict
import time, datetime
import random
import data2cv

def parse_argv(argv):
    opts, args = getopt.getopt(sys.argv[1:], "he:s:u:b:w:c:d:i:n:",
                               ['epoch', 'static','hidden_units',
                                'batch_size','window', 'active_function',
                                'dimension','inputdir','norm'])
    epochs = 30
    static = False
    hidden_units_str = '100_51'
    batch_size = 50
    window_size =3
    conv_non_linear = 'tanh' # active fuction
    dimension = 50
    inputdir = './'
    norm = 0
    for op, value in opts:
        if op == '-e':
            epochs = int(value)
        elif op == '-s':
            static = bool(int(value))
        elif op == '-u':
            hidden_units_str = value
        elif op == '-b':
            batch_size = int(value)
        elif op == '-w':
            window_size = int(value)
        elif op == '-a':
            conv_non_linear = value
        elif op == '-d':
            dimension = int(value)
        elif op == '-i':
            inputdir = value
        elif op =='-n':
            norm = int(value)
        elif op == '-h':
            #TODO
            #usage()
            sys.exit()
    return [epochs, static, hidden_units_str, batch_size, window_size, conv_non_linear, dimension, inputdir, norm]

def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)
def sgd_updates_adadelta(norm,params,cost,rho=0.95,epsilon=1e-6,norm_lim=9,word_vec_name='Words'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step = -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if norm == 1:
            if (param.get_value(borrow=True).ndim == 2) and param.name!='Words':
                col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
                desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
                scale = desired_norms / (1e-7 + col_norms)
                updates[param] = stepped_param * scale
            else:
                updates[param] = stepped_param
        elif norm == 0:
            updates[param] = stepped_param
        else:
            updates[param] = stepped_param
    return updates

def train_conv_net(train,
                    test,
                    U,
                    PF1,
                    PF2,
                    filter_hs=3,
                    conv_non_linear="tanh",
                    hidden_units=[100, 51],
                    shuffle_batch=True,
                    epochs=25,
                    sqr_norm_lim=9,
                    lr_decay=0.95,
                    static=False,
                    batch_size=50,
                    img_w=50,
                    pf_dim=5,
                    norm=0,
                    dropout_rate=[0.5],
                    directory='./',
                    activations_str=[],
                    borrow=True):
    # T.config.exception_verbosity='high'
    activations = []
    for act in activations_str:
        dropout_rate.append(0.5)
        if act.lower() == 'tanh':
            activations.append(Tanh)
        elif act.lower() == 'sigmoid':
            activations.append(Sigmoid)

    rng = np.random.RandomState(3435)
    img_h = len(train[0].sentences[0])# image height = 101
    filter_w = img_w # img_w = 50
    # All the sentence are transformed into a picture(2-d matrix). Pad with zeros.
    # The width of the picture equals the dimension of word embedding.
    # The height of the picture equals the number of tokens in the padded sentence.

    feature_maps = hidden_units[0]
    filter_shape = (feature_maps, 1, filter_hs, filter_w+pf_dim*2)
    # pool_size = (img_h-filter_hs+1, img_w-filter_w+1)

    # index = T.lscalar()
    x = T.imatrix('x')
    p1 = T.imatrix('pf1')
    p2 = T.imatrix('pf2')
    pool_size = T.imatrix('pos')
    y = T.ivector('y')

    Words = theano.shared(value=U, name="Words")
    PF1W = theano.shared(value=PF1, name="pf1w")
    PF2W = theano.shared(value=PF2, name="pf2w")

    zero_vec_tensor = T.vector()
    zero_vec = np.zeros(img_w)
    set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0,:], zero_vec_tensor))])

    zero_vec_tensor = T.vector()
    zero_vec_pf = np.zeros(pf_dim)
    set_zero_pf1 = theano.function([zero_vec_tensor], updates=[(PF1W, T.set_subtensor(PF1W[0,:], zero_vec_tensor))])
    set_zero_pf2 = theano.function([zero_vec_tensor], updates=[(PF2W, T.set_subtensor(PF2W[0,:], zero_vec_tensor))])

    # The first input layer
    # All the input tokens in a sentence are firstly transformed into vectors by looking up word embeddings.
    input_words = Words[x.flatten()].reshape((x.shape[0], 1, x.shape[1], Words.shape[1]))
    input_pf1 = PF1W[p1.flatten()].reshape((p1.shape[0], 1, p1.shape[1], pf_dim))
    input_pf2 = PF2W[p2.flatten()].reshape((p2.shape[0], 1, p2.shape[1], pf_dim))

    layer0_input = T.concatenate([input_words, input_pf1, input_pf2], axis=3)

    conv_layer = LeNetConvPoolLayer(rng, input=layer0_input,
                                    image_shape=(batch_size, 1, img_h, img_w+pf_dim*2),
                                    filter_shape=filter_shape, pool_size=pool_size,
                                    non_linear=conv_non_linear, max_window_len=3)
    layer1_input = conv_layer.output.flatten(2)

    # the number of hidden unit 0 equals to the features multiple the number of filter (100*1=100)
    hidden_units[0] = feature_maps*3
    classifier = MLPDropout(rng, input=layer1_input,
                            layer_sizes=hidden_units,
                            activations=activations,
                            dropout_rates=dropout_rate)
    params = classifier.params # sofmax parameters
    params += conv_layer.params # conv parameters

    if not static:  # if word vectors are allowed to change, add them as model parameters
        params += [Words]
        params += [PF1W]
        params += [PF2W]

    model_static = [(batch_size, 1, img_h, img_w+pf_dim*2), filter_shape, conv_non_linear, pool_size]
    model_static += [rng, hidden_units, activations, dropout_rate]

    #cost = classifier.negative_log_likelihood(y)
    p_y_given_x = classifier.p_y_given_x
    dropout_cost = classifier.dropout_negative_log_likelihood(y)
    grad_updates = sgd_updates_adadelta(norm, params, dropout_cost, lr_decay, 1e-6, sqr_norm_lim)

    #train data split
    #shuffle train dataset and assign to mini batches.
    np.random.seed(3435)
    #if dataset size is not a multiple of mini batches, replicate
    if len(train) % batch_size > 0:
        extra_data_num = batch_size - len(train) % batch_size
        rand_train = np.random.permutation(train)
        extra_data = rand_train[:extra_data_num]
        new_train = np.append(train, extra_data, axis=0)
    else:
        new_train = train

    new_train = np.random.permutation(new_train)
    n_train_batches = new_train.shape[0]/batch_size # batch number of train data

    [train_rels, train_nums, train_sents, train_poss, train_eposs] = bags_decompose(new_train)
    [test_rels, test_nums, test_sents, test_poss, test_eposs] = bags_decompose(test)

    test_size = 1
    test_input_words = Words[x.flatten()].reshape((x.shape[0], 1, x.shape[1], Words.shape[1]))
    test_input_pf1 = PF1W[p1.flatten()].reshape((p1.shape[0], 1, p1.shape[1], pf_dim))
    test_input_pf2 = PF2W[p2.flatten()].reshape((p2.shape[0], 1, p2.shape[1], pf_dim))

    test_layer0_input = T.concatenate([test_input_words, test_input_pf1, test_input_pf2], axis=3)
    test_layer0_output = conv_layer.predict(test_layer0_input, test_size, pool_size)
    test_layer1_input = test_layer0_output.flatten(2)
    p_y_given_x = classifier.predict_p(test_layer1_input)

    test_one = theano.function([x, p1, p2, pool_size], p_y_given_x)
    train_model_batch = theano.function([x, p1, p2, pool_size, y], dropout_cost, updates=grad_updates,)

    #start training over mini-batches
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print '... training start at  ' + str(now)
    epoch = 0
    while (epoch < epochs):
        for minibatch_index in np.random.permutation(range(n_train_batches)):
            batch_index = range(minibatch_index * batch_size, (minibatch_index + 1) * batch_size)
            batch_rels = [train_rels[m][0] for m in batch_index]
            batch_nums = [train_nums[m] for m in batch_index]
            batch_sents = [train_sents[m] for m in batch_index]
            batch_poss = [train_poss[m] for m in batch_index]
            batch_eposs = [train_eposs[m] for m in batch_index]
            batch_data = select_instance(batch_rels,
                            batch_nums,
                            batch_sents,
                            batch_poss,
                            batch_eposs,
                            test_one, img_h)
            # print batch_eposs
            batch_cost = train_model_batch(batch_data[0], batch_data[1], batch_data[2], batch_data[3], batch_data[4])
            set_zero(zero_vec)
            set_zero_pf1(zero_vec_pf)
            set_zero_pf2(zero_vec_pf)

        test_predict = predict_relation(test_rels, test_nums, test_sents, test_poss, test_eposs, test_one, img_h)
        test_pr = positive_evaluation(test_predict)
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        print str(now) + '\t epoch ' + str(epoch) + ' test set PR = [' + str(test_pr[0][-1]) + ' ' + str(test_pr[1][-1]) + ']'

        p = test_pr[0][-1]
        r = test_pr[1][-1]
        if p > 0.25 and r > 0.25:
            save_pr(directory+'test_pr_' + str(epoch) + '.txt', test_pr)
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            print str(now) + '\t epoch ' + str(epoch) + ' save PR result...'
        print '\n'
        epoch += 1

def save_model(file, params):
    f = open(file, 'w')
    cPickle.dump(params, f, -1)
    f.close()

def save_pr(file, pr):
    f = open(file, 'w')
    all_pre = pr[0]
    all_rec = pr[1]
    for i, p in enumerate(all_pre):
        f.write(str(p) + ' ' + str(all_rec[i]) + '\n')
    f.close()

def positive_evaluation(predict_results):
    predict_y = predict_results[0]
    predict_y_prob = predict_results[1]
    y_given = predict_results[2]

    positive_num = 0
    #find the number of positive examples
    for yi in range(y_given.shape[0]):
        if y_given[yi, 0] > 0:
            positive_num += 1
    # if positive_num == 0:
    #     positive_num = 1
    # sort prob
    index = np.argsort(predict_y_prob)[::-1]

    all_pre = [0]
    all_rec = [0]
    p_n = 0
    p_p = 0
    n_p = 0
    # print y_given.shape[0]
    for i in range(y_given.shape[0]):
        labels = y_given[index[i],:] # key given labels
        py = predict_y[index[i]] # answer

        if labels[0] == 0:
            # NA bag
            if py > 0:
                n_p += 1
        else:
            # positive bag
            if py == 0:
                p_n += 1
            else:
                flag = False
                for j in range(y_given.shape[1]):
                    if j == -1:
                        break
                    if py == labels[j]:
                        flag = True # true positive
                        break
                if flag:
                    p_p += 1
        if (p_p+n_p) == 0:
            precision = 1
        else:
            precision = float(p_p)/(p_p+n_p)
        recall = float(p_p)/positive_num
        if precision != all_pre[-1] or recall != all_rec[-1]:
            all_pre.append(precision)
            all_rec.append(recall)
    return [all_pre[1:], all_rec[1:]]

def select_instance(rels, nums, sents, poss, eposs, test_one, img_h):
    numBags = len(rels)
    x = np.zeros((numBags, img_h), dtype='int32')
    p1 = np.zeros((numBags, img_h), dtype='int32')
    p2 = np.zeros((numBags, img_h), dtype='int32')
    pool_size = np.zeros((numBags, 2), dtype='int32')
    y = np.asarray(rels, dtype='int32')

    for bagIndex, insNum in enumerate(nums):
        maxIns = 0
        maxP = -1
        if insNum > 1:
            for m in range(insNum):
                insPos = poss[bagIndex][m]
                insX = np.asarray(sents[bagIndex][m], dtype='int32').reshape((1, img_h))
                insPf1 = np.asarray(insPos[0], dtype='int32').reshape((1, img_h))
                insPf2 = np.asarray(insPos[1], dtype='int32').reshape((1, img_h))
                insPool = np.asarray(eposs[bagIndex][m], dtype='int32').reshape((1, 2))
                results = test_one(insX, insPf1, insPf2, insPool)
                tmpMax = results.max()
                if tmpMax > maxP:
                    maxIns = m
            # else:
            #     maxIns = 0
        x[bagIndex,:] = sents[bagIndex][maxIns]
        p1[bagIndex,:] = poss[bagIndex][maxIns][0]
        p2[bagIndex,:] = poss[bagIndex][maxIns][1]
        pool_size[bagIndex,:] = eposs[bagIndex][maxIns]
    return [x, p1, p2, pool_size, y]

def predict_relation(rels, nums, sents, poss, eposs, test_one, img_h):
    numBags = len(rels)
    predict_y = np.zeros((numBags), dtype='int32')
    predict_y_prob = np.zeros((numBags), dtype=theano.config.floatX)
    y = np.asarray(rels, dtype='int32')
    for bagIndex, insRel in enumerate(rels):
        insNum = nums[bagIndex]
        maxP = -1
        pred_rel_type = 0
        max_pos_p = -1
        positive_flag = False
        for m in range(insNum):
            insPos = poss[bagIndex][m]
            insX = np.asarray(sents[bagIndex][m], dtype='int32').reshape((1, img_h))
            insPf1 = np.asarray(insPos[0], dtype='int32').reshape((1, img_h))
            insPf2 = np.asarray(insPos[1], dtype='int32').reshape((1, img_h))
            insPool = np.asarray(eposs[bagIndex][m], dtype='int32').reshape((1, 2))
            results = test_one(insX, insPf1, insPf2, insPool)
            rel_type = results.argmax()
            if positive_flag and rel_type == 0:
                continue
            else:
                # at least one instance is positive
                tmpMax = results.max()
                if rel_type > 0:
                    positive_flag = True
                    if tmpMax > max_pos_p:
                        max_pos_p = tmpMax
                        pred_rel_type = rel_type
                else:
                    if tmpMax > maxP:
                        maxP = tmpMax
        if positive_flag:
            predict_y_prob[bagIndex] = max_pos_p
        else:
            predict_y_prob[bagIndex] = maxP

        predict_y[bagIndex] = pred_rel_type
    return [predict_y, predict_y_prob, y]

def bags_decompose(data_bags):
    bag_sent = [data_bag.sentences for data_bag in data_bags]
    bag_pos = [data_bag.positions for data_bag in data_bags]
    bag_num = [data_bag.num for data_bag in data_bags]
    bag_rel = [data_bag.rel for data_bag in data_bags]
    bag_epos = [data_bag.entitiesPos for data_bag in data_bags]
    return [bag_rel, bag_num, bag_sent, bag_pos, bag_epos]

if __name__ == "__main__":
    epochs, static, hidden_units_str, batch_size,\
    window_size, conv_non_linear, dimension, inputdir, norm = parse_argv(sys.argv[1:])


    hu_str = hidden_units_str.split('_')
    hidden_units = [int(hu_str[0])]
    activations = []
    for i in range(1,len(hu_str)-1,2):
        hidden_units.append(int(hu_str[i]))
        activations.append(hu_str[i+1])
    hidden_units.append(int(hu_str[-1]))

    if not os.path.isfile(inputdir+'/'+str(dimension)+'/Wv.p'):
        import dataset
        dataset.wv2pickle(inputdir+'/'+str(dimension)+'/wv.txt', dimension, inputdir+'/'+str(dimension)+'/Wv.p')

    resultdir = './e_'+str(epochs)+'_s_'+str(static)+'_u_'+\
                hidden_units_str+'_b_'+str(batch_size)+'_w_'+\
                str(window_size)+'_c_'+conv_non_linear+'_d_'+\
                str(dimension)+'_i_'+inputdir+'_n_'+str(norm)+'/'
    print 'result dir='+resultdir
    if not os.path.exists(resultdir):
        os.mkdir(resultdir)

    if not os.path.isfile(inputdir+'/test.p'):
        import dataset
        dataset.data2pickle(inputdir+'/test_filtered.data', inputdir+'/test.p')
    if not os.path.isfile(inputdir+'/train.p'):
        import dataset
        dataset.data2pickle(inputdir+'/train_filtered.data', inputdir+'/train.p')

    testData = cPickle.load(open(inputdir+'/test.p'))
    trainData = cPickle.load(open(inputdir+'/train.p'))
    # testData = testData[1:5]
    # trainData = trainData[1:15]
    tmp = inputdir.split('_')

    test = data2cv.make_idx_data_cv(testData, window_size, int(tmp[3]))
    train = data2cv.make_idx_data_cv(trainData, window_size, int(tmp[3]))

    print 'load Wv ...'
    Wv = cPickle.load(open(inputdir+'/'+str(dimension)+'/Wv.p'))

    rng = np.random.RandomState(3435)
    PF1 = np.asarray(rng.uniform(low=-1, high=1, size=[101, 5]), dtype=theano.config.floatX)
    padPF1 = np.zeros((1, 5))
    PF1 = np.vstack((padPF1, PF1))
    PF2 = np.asarray(rng.uniform(low=-1, high=1, size=[101, 5]), dtype=theano.config.floatX)
    padPF2 = np.zeros((1, 5))
    PF2 = np.vstack((padPF2, PF2))

    train_conv_net(train,
                    test,
                    Wv,
                    PF1,
                    PF2,
                    filter_hs=window_size,
                    conv_non_linear=conv_non_linear,
                    hidden_units=hidden_units,
                    activations_str=activations,
                    shuffle_batch=True,
                    epochs=epochs,
                    static=static,
                    directory=resultdir,
                    norm=norm,
                    batch_size=batch_size)

