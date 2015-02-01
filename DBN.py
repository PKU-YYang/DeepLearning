#-*- coding:utf-8 -*-

import os
import sys
import time
import re
import numpy
import rpy2.robjects as ro
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from rbm import RBM #唯一不一样的就是这里


# start-snippet-1
class DBN(object):

    def __init__(self,
                 numpy_rng,
                 theano_rng=None,
                 n_ins=784,
                 hidden_layers_sizes=[500, 500],
                 n_outs=10,
                 logLayer_weights_file=None,logLayer_bias_file=None,
                 hiddenLayer_weights_file=None,hiddenLayer_bias_file=None
    ):
        """This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the DBN

        :type hidden_layers_sizes: list of ints
        :param hidden_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network
        """

        self.sigmoid_layers = []
        self.rbm_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 17))


        self.x = T.matrix('x')
        self.y = T.ivector('y')


        for i in xrange(self.n_layers):

            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output #这个是前一层的输出

            #真正的deep learning network

            if hiddenLayer_weights_file is None:

                sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                            input=layer_input,
                                            n_in=input_size,
                                            n_out=hidden_layers_sizes[i],
                                            activation=T.nnet.sigmoid)

            else:

                print "Reading in the weights and bias of %d Hidden Layer" %(i+1)

                weights_filename="".join([str(i+1),"_hiddenLayer_W.csv"])
                bias_filename="".join([str(i+1),"_hiddenLayer_b.csv"])

                f=open(os.path.join(hiddenLayer_weights_file,weights_filename),"rb")
                data=numpy.loadtxt(f,delimiter=',',dtype=float)
                f.close()
                shared_hiddenLayer_W = theano.shared(numpy.asarray(data,dtype=theano.config.floatX),borrow=True)

                f=open(os.path.join(hiddenLayer_bias_file,bias_filename),"rb")
                data=numpy.loadtxt(f,delimiter=',',dtype=float)
                f.close()
                shared_hiddenLayer_b = theano.shared(numpy.asarray(data,dtype=theano.config.floatX),borrow=True)

                sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                            input=layer_input, #上一层的输出
                                            n_in=input_size, #上一层的大小
                                            n_out=hidden_layers_sizes[i],
                                            activation=T.nnet.sigmoid,
                                            W=shared_hiddenLayer_W,
                                            b=shared_hiddenLayer_b)



            self.sigmoid_layers.append(sigmoid_layer)

            #rbm只是用来train，本身不能算网络的组成部分
            self.params.extend(sigmoid_layer.params)

            # Construct an RBM that shared weights with this layer
            rbm_layer = RBM(numpy_rng=numpy_rng,
                            theano_rng=theano_rng,
                            input=layer_input, #这里说明rbm的输入可以是连续数值
                            n_visible=input_size,
                            n_hidden=hidden_layers_sizes[i],
                            W=sigmoid_layer.W, #这样rbm训练修改以后所有的w和b就是传入了mlp里面，并没有v
                            hbias=sigmoid_layer.b)
            self.rbm_layers.append(rbm_layer)

        # 最顶层放Logistic,有提供权重的时候就要去读入
        if logLayer_weights_file is None:

            self.logLayer = LogisticRegression(
                input=self.sigmoid_layers[-1].output,
                n_in=hidden_layers_sizes[-1],
                n_out=n_outs)
        else:

            print "Reading in the weights and bias of Logistic Layer \n"

            #把logistic layer的权重读入
            weights_filename=os.path.join(logLayer_weights_file,"logisticLayer_W.csv")
            bias_filename=os.path.join(logLayer_bias_file,"logisticLayer_b.csv")

            f=open(weights_filename,"rb")
            data=numpy.loadtxt(f,delimiter=',',dtype=float)
            f.close()
            shared_logLayer_W = theano.shared(numpy.asarray(data,dtype=theano.config.floatX),borrow=True)

            f=open(bias_filename,"rb")
            data=numpy.loadtxt(f,delimiter=',',dtype=float)
            f.close()
            shared_logLayer_b = theano.shared(numpy.asarray(data,dtype=theano.config.floatX),borrow=True)


            self.logLayer = LogisticRegression(
                input=self.sigmoid_layers[-1].output,
                n_in=hidden_layers_sizes[-1],
                n_out=n_outs,
                W=shared_logLayer_W,b=shared_logLayer_b
            )


        self.params.extend(self.logLayer.params)

        # 顶层的cost，也是整个网络的cost
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)

        #每个minibatch的错误
        self.errors = self.logLayer.errors(self.y)

    def pretraining_functions(self, train_set_x, batch_size, k):
        #返回的是training函数
        #返回训练每层的函数壳

        # index to a [mini]batch
        index = T.lscalar('index')
        learning_rate = T.scalar('lr')

        # number of minibatches
        n_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        batch_begin = index * batch_size
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for rbm in self.rbm_layers:

            #训练DBN的时候用的是CDK
            cost, updates = rbm.get_cost_updates(learning_rate,persistent=None, k=k)

            # compile the theano function
            fn = theano.function(
                inputs=[index, theano.Param(learning_rate, default=0.1)], #lr是这个参数的名字，不是learning_rate
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin:batch_end]
                }
            )
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on a
        batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                        the has to contain three pairs, `train`,
                        `valid`, `test` in this order, where each pair
                        is formed of two Theano variables, one for the
                        datapoints, the other for the labels
        :type batch_size: int
        :param batch_size: size of a minibatch
        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage

        '''

        # Generates a function `train` that implements one step of
        # finetuning, a function `validate` that computes the error on a
        # batch from the validation set, and a function `test` that
        # computes the error on a batch from the testing set

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        #这里的cost就是logsitc level的cost,熵
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam * learning_rate))

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
            }
        )

        test_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: test_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: test_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            }
        )

        valid_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: valid_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: valid_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            }
        )

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, valid_score, test_score

    def dbn_show_weights(self):

        weights=[]

        bias=[]

        #隐层
        for i in xrange(self.n_layers):

            weights.append(self.sigmoid_layers[i].W.get_value())
            bias.append(self.sigmoid_layers[i].b.get_value())

        #log layer
        weights.append(self.logLayer.W.get_value())
        bias.append(self.logLayer.b.get_value())

        return(weights,bias)

    def dbn_show_pretraining_labels(self,extendinput,i):
        #这个函数用来计算所有隐层的输出
        #input: n*m hidden_w: m*hidden_units hidden_b:hidden_units*1
        if i==0:
            return T.nnet.sigmoid(T.dot(extendinput, self.sigmoid_layers[i].W) + self.sigmoid_layers[i].b)
        else:
            return T.nnet.sigmoid(T.dot(self.dbn_show_pretraining_labels(extendinput,i-1),
                                        self.sigmoid_layers[i].W) + self.sigmoid_layers[i].b)

    def dbn_show_labels(self,extendinput):

        self.p_newy_given_x = T.nnet.softmax(T.dot(self.dbn_show_pretraining_labels(extendinput,self.n_layers-1)
                                                   , self.logLayer.W) + self.logLayer.b)

        self.p_newy = T.cast(T.argmax(self.p_newy_given_x, axis=1),'int32')
        #返回的是序号，axis=0是找每列中的最大值对应的行号，axis=1是找每行中得最大值对应的列号，计数是从0开始的
        #numpy.argmax

        return (self.p_newy,T.max(self.p_newy_given_x,axis=1))

    def dbn_show_all_labels(self,extendinput):
        #返回概率矩阵
        self.p_newy_given_x = T.nnet.softmax(T.dot(self.dbn_show_pretraining_labels(extendinput,self.n_layers-1)
                                                   , self.logLayer.W) + self.logLayer.b)

        return (self.p_newy_given_x)



def train_DBN(finetune_lr=0.1, pretraining_epochs=100,
             pretrain_lr=0.01,training_epochs=1000,
             dataset=[],n_in=29,n_out=2,
             batch_size=10,hidden_layers=[],
             cdk=1,
             weights_file=None,bias_file=None):

    datasets = load_data(dataset[0],dataset[1],dataset[2]) #一次读入三个dataset

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # numpy random generator
    numpy_rng = numpy.random.RandomState(89757)
    print '... building the model'
    # construct the Deep Belief Network
    dbn = DBN(numpy_rng=numpy_rng, n_ins=n_in,
              hidden_layers_sizes=hidden_layers,
              n_outs=n_out)


    #########################
    # PRETRAINING THE MODEL #
    #########################
    print '... getting the pretraining functions'
    pretraining_fns = dbn.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size,
                                                k=cdk) #cdk是在构建函数外壳的时候就确定好的了

    print '... pre-training the model'
    start_time = time.clock()

    ## 训练每个隐层
    for i in xrange(dbn.n_layers):
        # 每层有自己的epoch
        for epoch in xrange(pretraining_epochs):
            c = []
            for batch_index in xrange(n_train_batches):
                ########################################
                #真正训练每个隐层的地方
                ########################################
                c.append(pretraining_fns[i](index=batch_index, #训练第i层
                                            lr=pretrain_lr)) #sda直到这里才给corruption rate

            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print numpy.mean(c)

    end_time = time.clock()

    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing function for the model
    print '... getting the finetuning functions'
    train_fn, validate_model, test_model = dbn.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )

    print '... finetuning the model'
    #拉大下面三个参数都可以让finetun变得更长
    # early-stopping parameters
    patience = 30 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.    # wait this much longer when a new best is
                              # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatches before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            ################################
            # fine-tune the model         ##
            ################################

            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:

                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses)
                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%'
                    % (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = test_model()
                    test_score = numpy.mean(test_losses)
                    #因为这里的架构是train test分开的，所以test set上的error没有计算的必要，反正和validation一样
                    # print(('     epoch %i, minibatch %i/%i, test error of '
                    #        'best model %f %%') %
                    #       (epoch, minibatch_index + 1, n_train_batches,
                    #        test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(
        (
            'Optimization complete with best validation score of %f %%, '
            'obtained at iteration %i, '
            'with test performance %f %%'
        ) % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
    )
    print >> sys.stderr, ('The fine tuning code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time)/ 60.))

    #########把权重取出来
    if weights_file is not None:

        print("\n Extracting weights:")

        if not os.path.exists(weights_file):
            os.makedirs(weights_file)

        os.chdir(weights_file)


        weights,bias=dbn.dbn_show_weights()

        if len(weights)<>len(hidden_layers)+1:

            raise ValueError("The number of hidden layers is wrong!")


        #记录logistic layer的权重
        numpy.savetxt("logisticLayer_W.csv",weights.pop(),delimiter=",")
        numpy.savetxt("logisticLayer_b.csv",bias.pop(),delimiter=",")

        #记录hidden layer的权重
        for i in xrange(len(weights)):
            weights_filename=[str(i+1),"_hiddenLayer_W.csv"]
            bias_filename=[str(i+1),"_hiddenLayer_b.csv"]
            numpy.savetxt("".join(weights_filename),weights[i],delimiter=",")
            numpy.savetxt("".join(bias_filename),bias[i],delimiter=",")

        print "Weights and bias have been saved! \n"

def extend_DBN(newx=None,newy=None,
               weights_file=None,bias_file=None,
               n_in=None,n_out=None,hidden_layers=[100,100]):

    numpy_rng = numpy.random.RandomState(89757)

    #普通的logistic model里面，每层的weight是在extend函数里面读入并且送入的，函数送的是具体的位置
    # 在sda model里面，每层的weight是在sda自己的init函数里读入的，函数送的是weights所在的文件夹
    print "Reading in the new input\n"

    f=open(newx,"rb")
    newdata=numpy.loadtxt(f,delimiter=',',dtype=float,skiprows=1)
    f.close()
    newinput = theano.shared(numpy.asarray(newdata,dtype=theano.config.floatX),borrow=True)


    dbn = DBN(
        numpy_rng=numpy_rng,
        n_ins=n_in,
        hidden_layers_sizes=hidden_layers,
        n_outs=n_out,
        logLayer_weights_file=weights_file,logLayer_bias_file=bias_file,
        hiddenLayer_weights_file=weights_file,hiddenLayer_bias_file=bias_file
    )


    print "Extending on the new data"

    #compile the function,this returns the label and its prob
    extend_model = theano.function(
    inputs=[],
    outputs=dbn.dbn_show_labels(newinput)
     #如果想要返回一个用theano计算的函数，那么一定要额外在主函数里建一个专门返回子函数的函数
    )

    #this returns the prob of being all labels
    extend_model_all_results = theano.function(
    inputs=[],
    outputs=dbn.dbn_show_all_labels(newinput)
     #如果想要返回一个用theano计算的函数，那么一定要额外在主函数里建一个专门返回子函数的函数
    )


    labels,prob=extend_model()
    prob_matrix=extend_model_all_results()
    #print(prob_matrix)
    fmt = ",".join(["%i"] + ["%f"])
    numpy.savetxt(newy[0],zip(labels,prob),fmt=fmt,delimiter=',') #save the label
    numpy.savetxt(newy[1],numpy.asarray(prob_matrix),delimiter=',')
    print "New label has been generated!"





def main(argv=sys.argv):

    if sys.argv[1]=="DeepLearning-Train":

        #######################################
        #### call R to preprocessing        ###
        #######################################

        code_dir=os.getcwd()

        dpdata_address=os.path.split(os.path.dirname(sys.argv[2]))[-1]

        r=ro.r

        wd="".join(['setwd(\'',os.path.join(os.path.split(os.getcwd())[0],dpdata_address),'\')']).replace("\\","/") #这个是windows特有的双\\地址

        r(wd) #这句会改变python的当前路径

        r.source(os.path.join(code_dir,'DeepLearning-PreProcessing.R'))


        ro.globalenv['train_test_ratio']=float(sys.argv[3])

        ro.globalenv['training_data']=sys.argv[2]

        ro.globalenv['label_no']=int(sys.argv[4])

        inout=r('train_preprocessing(train_test_ratio,training_data,label_no)')

        no_input=int(inout[0])
        no_output=int(inout[1])

        #########################################

        data_header=os.path.split(sys.argv[2])[0]
        datasetname=[os.path.join(data_header,"DP_train.csv"),os.path.join(data_header,"DP_valid.csv"),os.path.join(data_header,"DP_valid.csv")]

        training_learning_rate=float(sys.argv[5])

        tuning_learning_rate=float(sys.argv[6])

        training_epochs=int(sys.argv[7])

        tuning_epochs=int(sys.argv[8])

        batchsize=int(sys.argv[9])

        hiddenlayers=eval(sys.argv[10])

        cdk=int(sys.argv[11])

        train_DBN(finetune_lr=tuning_learning_rate, pretraining_epochs=training_epochs,
                    pretrain_lr=training_learning_rate, training_epochs=tuning_epochs,
                    dataset=datasetname,
                    n_in=no_input,n_out=no_output,
                    batch_size=batchsize, hidden_layers=hiddenlayers,
                    cdk=cdk,
                    weights_file=os.path.join(os.getcwd(),'DP_classifier'),
                    bias_file=os.path.join(os.getcwd(),'DP_classifier'))


    elif sys.argv[1]=="DeepLearning-Extend":
        #####################################
        ### Extending the model on new data #
        #####################################

        #注意extend的时候要看清楚当前读入的weights是哪个pl定义下的,weights的clasifier文件名前面没有pl的名字

        ro.globalenv['training_data']=sys.argv[2]

        ro.globalenv['extending_data']=sys.argv[3]

        ro.globalenv['non_feature']=int(sys.argv[4])

        hiddenlayers=eval(sys.argv[5])


        print ('\nCalling R to pre-processing the data\n')


        code_dir=os.getcwd()

        dpdata_address=os.path.split(os.path.dirname(sys.argv[2]))[-1]

        r=ro.r

        wd="".join(['setwd(\'',os.path.join(os.path.split(os.getcwd())[0],dpdata_address),'\')']).replace("\\","/")

        r(wd) #这句会改变python的当前路径

        r.source(os.path.join(code_dir,'DeepLearning-PreProcessing.R'))

        #normalize the data using R
        inout=r('extend_preprocessing(training_data,extending_data, non_feature)')

        no_input=int(inout[0])
        no_output=int(inout[1])

        filehead=os.path.join(os.getcwd(),'DP_classifier')

        data_header=os.path.split(sys.argv[2])[0]
        resulthead=[os.path.join(data_header,"DP_result.csv"),os.path.join(data_header,"DP_result_all.csv")] #results name


        extend_DBN(newx=os.path.join(data_header,"DP_extend.csv"),
                    newy=resulthead,
                    weights_file=filehead,
                    bias_file=filehead,
                    n_in=no_input,n_out=no_output,hidden_layers=hiddenlayers)


if __name__ == '__main__':

    main()

    #python DBN.py DeepLearning-Train ../dpdata/m_train.csv 0.9 11 0.1 0.1 2 2 1 [100]*2 2

    #python DBN.py DeepLearning-Extend ../dpdata/m_train.csv m_extend.csv 11 [100]*2