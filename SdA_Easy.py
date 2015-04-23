#-*- coding:utf-8 -*-

import os
import sys
import time
import numpy
import theano
import theano.tensor as T

from theano.tensor.shared_randomstreams import RandomStreams


def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)

def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)

def Tanh(x):
    y = T.tanh(x)
    return(y)

def load_data(trainset,validset,testset):

    #分别读入三个文件并share他们

    data=numpy.loadtxt(validset, delimiter=',', dtype=float, skiprows=1)
    valid_set=(data[:,:-1],data[:,-1])

    data=numpy.loadtxt(testset, delimiter=',', dtype=float, skiprows=1)
    test_set=(data[:,:-1],data[:,-1])

    data=numpy.loadtxt(trainset, delimiter=',', dtype=float, skiprows=1)
    train_set=(data[:,:-1],data[:,-1])

    def shared_dataset(data_xy, borrow=True):

        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)

        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

class LogisticRegression(object): #As for the logistic regression we will
                                  # first define the log-likelihood and then
                                  # the loss function as being the negative log-likelihood.

    def __init__(self, input, n_in, n_out, W=None,b=None):



        if W is None:
            self.W = theano.shared(
                value=numpy.zeros(
                    (n_in, n_out), #其实w的每一列就是一个分类器，softmax就是归一化一个输入在每个类的分类器上的得分
                    dtype=theano.config.floatX
                ),
                name='W',
                borrow=True
            )
        else:
            self.W = W

        if b is None:
            self.b = theano.shared(
                value=numpy.zeros(
                    (n_out,),
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )
        else:
            self.b=b


        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):

        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y]) #把0,1,2,3..100中应该是的那个label的概率取出来

    def errors(self, y):

        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )

        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

    def show_labels(self,extendinput):

        self.p_newy_given_x = T.nnet.softmax(T.dot(extendinput, self.W) + self.b)

        self.p_newy = T.cast(T.argmax(self.p_newy_given_x, axis=1),'int32')

        return (self.p_newy,T.max(self.p_newy_given_x, axis=1))

    def show_weights(self):

        return((self.W.get_value(),self.b.get_value()))

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):

        self.input = input

        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b
        #hidden layer的输出是连续数值，在0,1之间
        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]

class dA(object):


    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        input=None,
        n_visible=784,
        n_hidden=500,
        W=None,
        bhid=None,
        bvis=None
    ):

        self.n_visible = n_visible
        self.n_hidden = n_hidden


        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))


        if not W:

            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if not bhid:
            bhid = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )

        self.W = W

        self.b = bhid

        self.b_prime = bvis

        self.W_prime = self.W.T
        self.theano_rng = theano_rng

        if input is None:

            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]
    # end-snippet-1

    def get_corrupted_input(self, input, corruption_level):

        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input

    def get_hidden_values(self, input):

        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):

        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, corruption_level, learning_rate):


        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)

        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)

        cost = T.mean(L)


        gparams = T.grad(cost, self.params) #pretrain的时候是基于信息熵的，没有用到label，尽量保证输出和输入要一样
                                            #是一种un-supervised的方法，和rbm一样
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        return (cost, updates)

class SdA(object):

    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        n_ins=784,
        hidden_layers_sizes=None,
        n_outs=10,
        corruption_levels=None,
        logLayer_weights_file=None, logLayer_bias_file=None,
        hiddenLayer_weights_file=None, hiddenLayer_bias_file=None,
        actfunc=Sigmoid
    ):

        self.sigmoid_layers = []
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 17)) #一个除了89757改变随机的地方

        self.x = T.matrix('x')
        self.y = T.ivector('y')

        for i in xrange(self.n_layers):
            if i == 0:
                input_size = n_ins
                layer_input = self.x
            else:
                input_size = hidden_layers_sizes[i - 1]
                layer_input = self.sigmoid_layers[-1].output

            #sigmoid layer用来最后的sda

            if hiddenLayer_weights_file is None:

                sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                            input=layer_input, #上一层的输出
                                            n_in=input_size, #上一层的大小
                                            n_out=hidden_layers_sizes[i],
                                            activation=actfunc)
            else:

                print "Reading in the weights and bias of %d Hidden Layer" %(i+1)

                weights_filename = "".join([str(i+1), "_hiddenLayer_W.csv"])
                bias_filename = "".join([str(i+1), "_hiddenLayer_b.csv"])

                f = open(os.path.join(hiddenLayer_weights_file, weights_filename), "rb")
                data = numpy.loadtxt(f, delimiter=',', dtype=float)
                f.close()
                shared_hiddenLayer_W = theano.shared(numpy.asarray(data, dtype=theano.config.floatX), borrow=True)

                f = open(os.path.join(hiddenLayer_bias_file, bias_filename), "rb")
                data = numpy.loadtxt(f, delimiter=',', dtype=float)
                f.close()
                shared_hiddenLayer_b = theano.shared(numpy.asarray(data, dtype=theano.config.floatX), borrow=True)

                sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                            input=layer_input, #上一层的输出
                                            n_in=input_size, #上一层的大小
                                            n_out=hidden_layers_sizes[i],
                                            activation=actfunc,
                                            W=shared_hiddenLayer_W,
                                            b=shared_hiddenLayer_b)

            self.sigmoid_layers.append(sigmoid_layer)

            # params才是整个model真正的参数
            self.params.extend(sigmoid_layer.params)

            # da 用来pretraining
            dA_layer = dA(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=hidden_layers_sizes[i],
                          W=sigmoid_layer.W,
                          bhid=sigmoid_layer.b)
            self.dA_layers.append(dA_layer)

        # 最后加一层logistic，有提供权重的时候就要去读入
        if logLayer_weights_file is None:

            self.logLayer = LogisticRegression(
                input=self.sigmoid_layers[-1].output,
                n_in=hidden_layers_sizes[-1],
                n_out=n_outs
            )

        else:

            print "Reading in the weights and bias of Logistic Layer \n"

            # 把logistic layer的权重读入
            weights_filename = os.path.join(logLayer_weights_file, "logisticLayer_W.csv")
            bias_filename = os.path.join(logLayer_bias_file, "logisticLayer_b.csv")

            f = open(weights_filename,"rb")
            data = numpy.loadtxt(f, delimiter=',', dtype=float)
            f.close()
            shared_logLayer_W = theano.shared(numpy.asarray(data, dtype=theano.config.floatX), borrow=True)

            f = open(bias_filename, "rb")
            data=numpy.loadtxt(f, delimiter=',', dtype=float)
            f.close()
            shared_logLayer_b = theano.shared(numpy.asarray(data, dtype=theano.config.floatX), borrow=True)

            self.logLayer = LogisticRegression(
                input=self.sigmoid_layers[-1].output,
                n_in=hidden_layers_sizes[-1],
                n_out=n_outs,
                W=shared_logLayer_W,
                b=shared_logLayer_b
            )

        self.params.extend(self.logLayer.params)

        # logistic专用的cost函数
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)

        # 实用的错误函数
        self.errors = self.logLayer.errors(self.y)

    def pretraining_functions(self, train_set_x, batch_size):
        # 构建train函数的外壳
        index = T.lscalar('index')
        corruption_level = T.scalar('corruption')
        learning_rate = T.scalar('lr')

        batch_begin = index * batch_size
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for dA in self.dA_layers:

            # da的cost是一个特殊的cross-entropy函数
            cost, updates = dA.get_cost_updates(corruption_level,
                                                learning_rate)

            # train函数的外壳
            fn = theano.function(
                inputs=[
                    index,
                    theano.Param(corruption_level, default=0.2),
                    theano.Param(learning_rate, default=0.1)
                ],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin: batch_end]
                }
            )
            # print pp(fn)
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, datasets, batch_size, learning_rate):

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        # logistic专用的cost函数
        gparams = T.grad(self.finetune_cost, self.params)

        updates = [
            (param, param - gparam * learning_rate)
            for param, gparam in zip(self.params, gparams)
        ]

        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: train_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='train'
        )

        valid_score_i = theano.function(
            [],
            self.errors,
            givens={
                self.x: valid_set_x,
                self.y: valid_set_y
            },
            name='valid'
        )

        def valid_score(): #调用valid_score函数可以返回所有Batch上的self.errors
            return valid_score_i()


        return train_fn, valid_score # train_fn是最外层训练函数的壳子

    def sda_show_weights(self):

        weights = []

        bias = []

        for i in xrange(self.n_layers):

            weights.append(self.sigmoid_layers[i].W.get_value())
            bias.append(self.sigmoid_layers[i].b.get_value())

        # 隐层的取完把最后一层加入
        weights.append(self.logLayer.W.get_value())
        bias.append(self.logLayer.b.get_value())

        return weights, bias

    def sda_show_pretraining_labels(self, extendinput, i):
        # 这个函数用来计算所有隐层的输出，这是一个中间函数
        # input: n*m hidden_w: m*hidden_units hidden_b:hidden_units*1
        if i==0:
            return T.nnet.sigmoid(T.dot(extendinput, self.sigmoid_layers[i].W) + self.sigmoid_layers[i].b)
        else:
            return T.nnet.sigmoid(T.dot(self.sda_show_pretraining_labels(extendinput,i-1),
                                        self.sigmoid_layers[i].W) + self.sigmoid_layers[i].b)
                                        # 这里的layer如果有新的读入会用新的读入

    def sda_show_labels(self, extendinput):
        # 计算sigmoid的输出
        self.p_newy_given_x = T.nnet.softmax(T.dot(self.sda_show_pretraining_labels(extendinput,self.n_layers-1)
                                                   , self.logLayer.W) + self.logLayer.b)

        self.p_newy = T.cast(T.argmax(self.p_newy_given_x, axis=1), 'int32')
        # 返回的是序号，axis=0是找每列中的最大值对应的行号，axis=1是找每行中得最大值对应的列号，计数是从0开始的
        # numpy.argmax

        return self.p_newy, T.max(self.p_newy_given_x,axis=1)

    def sda_show_all_labels(self, extendinput):
        # 返回概率矩阵
        self.p_newy_given_x = T.nnet.softmax(T.dot(self.sda_show_pretraining_labels(extendinput, self.n_layers-1)
                                                   , self.logLayer.W) + self.logLayer.b)

        return self.p_newy_given_x


def train_SdA(finetune_lr=0.01, pretraining_epochs=30,
              pretrain_lr=0.01, training_epochs=1000,
              dataset=None, n_in=29,n_out=2,
              batch_size=100, hidden_layers=None,
              corruption_levels=None,
              weights_file=None,  # where to save the new weights
              weights_initial=None,bias_initial=None
              ):

    datasets = load_data(dataset[0],dataset[1],dataset[2])

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size

    numpy_rng = numpy.random.RandomState(89757)
    print '... building the model'

    # 定义整个架构和cost函数
    sda = SdA(
        numpy_rng=numpy_rng,
        n_ins=n_in,
        hidden_layers_sizes=hidden_layers,
        n_outs=n_out,
        logLayer_weights_file=weights_initial,logLayer_bias_file=bias_initial,
        hiddenLayer_weights_file=weights_initial,hiddenLayer_bias_file=bias_initial
    )
    print '... getting the pretraining functions'
    pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size)

    print '... pre-training the model'
    start_time = time.clock()

    # 最外面，训练每个隐层
    for i in xrange(sda.n_layers):

        for epoch in xrange(pretraining_epochs): # 对于每个层，有自己的每个epoch

            c = []
            for batch_index in xrange(n_train_batches):  # 在每个epoch里面，训练整个测试集
                ##########################################
                #          pre-train the model           #
                ##########################################

                c.append(pretraining_fns[i](index=batch_index,
                         corruption=corruption_levels[i],  # corruption_level[i]就是在具体调用train函数的时候才传进去的
                         lr=pretrain_lr))
                # print pp(pretraining_fns[i])
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print numpy.mean(c)

    end_time = time.clock()

    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    print '... getting the finetuning functions'
    train_fn, validate_model = sda.build_finetune_functions(
        datasets=datasets, # valid和test dataset其实是在这里传入的
        batch_size=batch_size,
        learning_rate=finetune_lr
    )

    print '... finetunning the model'

    patience = 200 * n_train_batches
    patience_increase = 10.
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience / 2)


    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            ###########################################
            #                fine-tune the model      #
            ###########################################

            minibatch_avg_cost = train_fn(minibatch_index) # 最后train最顶层
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:

                this_validation_loss = validate_model()

                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))


                if this_validation_loss < best_validation_loss:

                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold  # 只有在效果有一定提升的情况下才会变大patience
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    print >> sys.stderr, (
                          'best validation score of %f %%' % (best_validation_loss * 100.))

                    weights,bias=sda.sda_show_weights()
                    # 想返回theano内部的什么东西，在类里面添加那个方法，然后再函数外面调用那个函数


            if patience <= iter: # iter会一直变大，patience在每次得到新的结果时也会变大
                done_looping = True
                break

    if weights_file is not None:

        print("\n Extracting weights:")

        if not os.path.exists(weights_file):
            os.makedirs(weights_file)

            os.chdir(weights_file)


        if len(weights)<>len(hidden_layers)+1:

            raise ValueError("The number of hidden layers is wrong!")


        # 记录logistic layer的权重,存在最后的是logistic
        numpy.savetxt("logisticLayer_W.csv", weights.pop(), delimiter=",")
        numpy.savetxt("logisticLayer_b.csv", bias.pop(), delimiter=",")

                        # 记录hidden layer的权重
        for i in xrange(len(weights)):
            weights_filename=[str(i+1), "_hiddenLayer_W.csv"]
            bias_filename=[str(i+1), "_hiddenLayer_b.csv"]
            # scipy.misc.imsave("".join([str(i+1), "_hiddenLayer_W.jpg"]), numpy.transpose(weights[i]))
            # [n_out,n_in]
            numpy.savetxt("".join(weights_filename), weights[i], delimiter=",")
            numpy.savetxt("".join(bias_filename), bias[i], delimiter=",")

        print "Weights and bias have been saved! \n"

    end_time = time.clock()
    print(
        (
            'Optimization complete with best validation score of %f %%, '
            'on iteration %i, '
        )
        % (best_validation_loss * 100., best_iter + 1)
    )
    print >> sys.stderr, ('The training code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.) +
                          'best validation score of %f %%' % (best_validation_loss * 100.))



def test_SdA(newx=None, newy=None,
             weights_file=None, bias_file=None,
             n_in=None, n_out=None, hidden_layers=None):

    numpy_rng = numpy.random.RandomState(89757)

    # 普通的logistic model里面，每层的weight是在extend函数里面读入并且送入的，函数送的是具体的位置
    # 在sda model里面，每层的weight是在sda自己的init函数里读入的，函数送的是weights所在的文件夹
    print "Reading in the new input\n"


    newdata=numpy.loadtxt(newx, delimiter=',', dtype=float, skiprows=1)
    newinput = theano.shared(numpy.asarray(newdata, dtype=theano.config.floatX), borrow=True)


    sda = SdA(
        numpy_rng=numpy_rng,
        n_ins=n_in,
        hidden_layers_sizes=hidden_layers,
        n_outs=n_out,
        logLayer_weights_file=weights_file, logLayer_bias_file=bias_file,
        hiddenLayer_weights_file=weights_file, hiddenLayer_bias_file=bias_file
    )

    print "Predicting on the new data"

    # compile the function,this returns the label and its prob
    extend_model = theano.function(
    inputs=[],
    outputs=sda.sda_show_labels(newinput)
    # 如果想要返回一个用theano计算的函数，那么一定要额外在主函数里建一个专门返回子函数的函数
    )

    # this returns the prob of being all labels
    extend_model_all_results = theano.function(
    inputs=[],
    outputs=sda.sda_show_all_labels(newinput)
     # 如果想要返回一个用theano计算的函数，那么一定要额外在主函数里建一个专门返回子函数的函数，就比如现在
    )

    labels,prob=extend_model()
    prob_matrix=extend_model_all_results()
    fmt = ",".join(["%i"] + ["%f"])
    numpy.savetxt(newy[0], zip(labels,prob), fmt=fmt, delimiter=',') # save the label
    numpy.savetxt(newy[1], numpy.asarray(prob_matrix), delimiter=',')
    print "New label and prob have been generated!"


def main(argv=sys.argv):


    if sys.argv[1] == "Train":

        if sys.argv[2] == 'random':
            new_initial_folder = None
        else:
            new_initial_folder = sys.argv[2]

        datasetname=["../dpdata/DP_train.csv","../dpdata/DP_valid.csv","../dpdata/DP_valid.csv"]

        train_SdA(finetune_lr=0.1, pretraining_epochs=3,
                  pretrain_lr=0.1, training_epochs=2,
                  dataset=datasetname,
                  n_in=10, n_out=3,
                  batch_size=1, hidden_layers=[10, 10],
                  corruption_levels=[0.3, 0.3],
                  weights_file=sys.argv[3], # where to save the weights                #如果有指定的初始权值，那么就要换成新的名字
                  weights_initial=new_initial_folder,bias_initial=new_initial_folder)  #如果有特定的初始权值，就从这里读入


    elif sys.argv[1] == 'Test':

        filehead = sys.argv[2]

        resulthead = [os.path.join(filehead,"DP_result.csv"),os.path.join(filehead,"DP_result_all.csv")]

        test_SdA(newx="../LCG_paper/DP_test.csv", # 不可以有label
                 newy=resulthead,
                 weights_file=filehead,
                 bias_file=filehead,
                 n_in=30,n_out=2,
                 hidden_layers=[128,1024,1024,128])

    else:
        raise ValueError("Please Input \'Test\' or \'Train\' ")


if __name__ == '__main__':


    main()


   #python SdALCG_v1.py Train random ../dpdata/0_weights  #random或者换成读取权重的地方

   #python SdALCG_v1.py Test ../dpdata/0_weights

