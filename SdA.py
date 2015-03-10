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
from dA import dA


# start-snippet-1
class SdA(object):

    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        n_ins=784,
        hidden_layers_sizes=[500, 500],
        n_outs=10,
        corruption_levels=[0.1, 0.1],
        logLayer_weights_file=None,logLayer_bias_file=None,
        hiddenLayer_weights_file=None,hiddenLayer_bias_file=None
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
                                            activation=T.nnet.sigmoid)
            else:

                #read in hidden layer weights

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

            #params才是整个model真正的参数
            self.params.extend(sigmoid_layer.params)

            #da 用来pretraining
            dA_layer = dA(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=hidden_layers_sizes[i],
                          W=sigmoid_layer.W,
                          bhid=sigmoid_layer.b)
            self.dA_layers.append(dA_layer)

        #最后加一层logistic，有提供权重的时候就要去读入
        if logLayer_weights_file is None:

            self.logLayer = LogisticRegression(
                input=self.sigmoid_layers[-1].output,
                n_in=hidden_layers_sizes[-1],
                n_out=n_outs
            )

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

        #logistic专用的cost函数
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)

        #实用的错误函数
        self.errors = self.logLayer.errors(self.y)

    def pretraining_functions(self, train_set_x, batch_size):
        #构建train函数的外壳
        index = T.lscalar('index')
        corruption_level = T.scalar('corruption')
        learning_rate = T.scalar('lr')

        batch_begin = index * batch_size
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for dA in self.dA_layers:

            #da的cost是一个特殊的cross-entropy函数
            cost, updates = dA.get_cost_updates(corruption_level,
                                                learning_rate)

            #train函数的外壳
            fn = theano.function(
                inputs=[
                    index,
                    theano.Param(corruption_level, default=0.2),
                    theano.Param(learning_rate, default=0.1)
                ],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin: batch_end] #存入cpu
                }
            )
            #print pp(fn)
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

        #logistic专用的cost函数
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
            },
            name='test'
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
            },
            name='valid'
        )

        def valid_score(): #调用valid_score函数可以返回所有Batch上的self.errors
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, valid_score, test_score #train_fn是最外层训练函数的壳子

    def sda_show_weights(self):

        weights=[]

        bias=[]

        for i in xrange(self.n_layers):

            weights.append(self.sigmoid_layers[i].W.get_value())
            bias.append(self.sigmoid_layers[i].b.get_value())

        #隐层的取完把最后一层加入
        weights.append(self.logLayer.W.get_value())
        bias.append(self.logLayer.b.get_value())

        return(weights,bias)

    def sda_show_pretraining_labels(self, extendinput,i):
        #这个函数用来计算所有隐层的输出，这是一个中间函数
        #input: n*m hidden_w: m*hidden_units hidden_b:hidden_units*1
        if i==0:
            return T.nnet.sigmoid(T.dot(extendinput, self.sigmoid_layers[i].W) + self.sigmoid_layers[i].b)
        else:
            return T.nnet.sigmoid(T.dot(self.sda_show_pretraining_labels(extendinput,i-1),
                                        self.sigmoid_layers[i].W) + self.sigmoid_layers[i].b)
                                        #这里的layer如果有新的读入会用新的读入

    def sda_show_labels(self,extendinput):
        #计算sigmoid的输出
        self.p_newy_given_x = T.nnet.softmax(T.dot(self.sda_show_pretraining_labels(extendinput,self.n_layers-1)
                                                   , self.logLayer.W) + self.logLayer.b)

        self.p_newy = T.cast(T.argmax(self.p_newy_given_x, axis=1),'int32')
        #返回的是序号，axis=0是找每列中的最大值对应的行号，axis=1是找每行中得最大值对应的列号，计数是从0开始的
        #numpy.argmax

        return (self.p_newy,T.max(self.p_newy_given_x,axis=1))

    def sda_show_all_labels(self,extendinput):
        #返回概率矩阵
        self.p_newy_given_x = T.nnet.softmax(T.dot(self.sda_show_pretraining_labels(extendinput,self.n_layers-1)
                                                   , self.logLayer.W) + self.logLayer.b)

        return (self.p_newy_given_x)


def train_SdA(finetune_lr=0.01, pretraining_epochs=30,
             pretrain_lr=0.01, training_epochs=1000,
             dataset=[], n_in=29,n_out=2,
             batch_size=100, hidden_layers=[],
             corruption_levels=[],
             newx=None,newy=None,
             weights_file=None,bias_file=None,  #where to save the new weights
             weights_initial=None,bias_initial=None
             ):

    datasets = load_data(dataset[0],dataset[1],dataset[2])

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size

    numpy_rng = numpy.random.RandomState(89757)
    print '... building the model'

    #定义整个架构和cost函数
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

    #最外面，训练每个隐层
    for i in xrange(sda.n_layers):

        for epoch in xrange(pretraining_epochs): #对于每个层，有自己的每个epoch

            c = []
            for batch_index in xrange(n_train_batches):  #在每个epoch里面，训练整个测试集
                ##########################################
                #          pre-train the model           #
                ##########################################

                c.append(pretraining_fns[i](index=batch_index,
                         corruption=corruption_levels[i],  #corruption_level[i]就是在具体调用train函数的时候才传进去的
                         lr=pretrain_lr))
                #print pp(pretraining_fns[i])
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print numpy.mean(c)

    end_time = time.clock()

    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    print '... getting the finetuning functions'
    train_fn, validate_model, test_model = sda.build_finetune_functions(
        datasets=datasets, #valid和test dataset其实是在这里传入的
        batch_size=batch_size,
        learning_rate=finetune_lr
    )

    print '... finetunning the model'

    patience = 30 * n_train_batches
    patience_increase = 2.
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

            minibatch_avg_cost = train_fn(minibatch_index) #最后train最顶层
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model() #这里返回的是每个batch上的cost
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))


                if this_validation_loss < best_validation_loss:

                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold  #只有在效果有一定提升的情况下才会变大patience
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    test_losses = test_model() #只有在validation上最小才会给机会在test上测试
                    test_score = numpy.mean(test_losses)
                    # print(('     epoch %i, minibatch %i/%i, test error of '
                    #        'best model %f %%') %
                    #       (epoch, minibatch_index + 1, n_train_batches,
                    #        test_score * 100.))

            if patience <= iter: #iter会一直变大，patience在每次得到新的结果时也会变大
                done_looping = True
                break

    end_time = time.clock()
    print(
        (
            'Optimization complete with best validation score of %f %%, '
            'on iteration %i, '
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
    )
    print >> sys.stderr, ('The training code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


    if weights_file is not None:

        print("\n Extracting weights:")

        if not os.path.exists(weights_file):
            os.makedirs(weights_file)

        os.chdir(weights_file)

        weights,bias=sda.sda_show_weights() #想返回theano内部的什么东西，在类里面添加那个方法，然后再函数外面调用那个函数

        if len(weights)<>len(hidden_layers)+1:

            raise ValueError("The number of hidden layers is wrong!")


        #记录logistic layer的权重,存在最后的是logistic
        numpy.savetxt("logisticLayer_W.csv",weights.pop(),delimiter=",")
        numpy.savetxt("logisticLayer_b.csv",bias.pop(),delimiter=",")

        #记录hidden layer的权重
        for i in xrange(len(weights)):
            weights_filename=[str(i+1),"_hiddenLayer_W.csv"]
            bias_filename=[str(i+1),"_hiddenLayer_b.csv"]
            numpy.savetxt("".join(weights_filename),weights[i],delimiter=",")
            numpy.savetxt("".join(bias_filename),bias[i],delimiter=",")

        print "Weights and bias have been saved! \n"




    if newx is None:
        print "There is no extending data \n"
    else:
        #print "Now extending on the new data \n"

        #read in data
        f=open(newx,"rb")
        newdata=numpy.loadtxt(f,delimiter=',',dtype=float,skiprows=1)
        f.close()

        #share it to theano
        newinput = theano.shared(numpy.asarray(newdata,dtype=theano.config.floatX),borrow=True)


        #compile the function
        extend_model = theano.function(
        inputs=[],
        outputs=sda.sda_show_labels(newinput)
         #如果想要返回一个用theano计算的函数，那么一定要额外在主函数里建一个专门返回子函数的函数
        )

        labels,prob=extend_model()
        #print(labels)
        fmt = ",".join(["%i"] + ["%f"])
        numpy.savetxt(newy,zip(labels,prob),fmt=fmt,delimiter=',')
        #print "New label has been generated! \n"



def extend_SdA(newx=None,newy=None,
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


    sda = SdA(
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
    outputs=sda.sda_show_labels(newinput)
     #如果想要返回一个用theano计算的函数，那么一定要额外在主函数里建一个专门返回子函数的函数
    )

    #this returns the prob of being all labels
    extend_model_all_results = theano.function(
    inputs=[],
    outputs=sda.sda_show_all_labels(newinput)
     #如果想要返回一个用theano计算的函数，那么一定要额外在主函数里建一个专门返回子函数的函数，就比如现在
    )



    labels,prob=extend_model()
    prob_matrix=extend_model_all_results()
    #print(prob_matrix)
    fmt = ",".join(["%i"] + ["%f"])
    numpy.savetxt(newy[0],zip(labels,prob),fmt=fmt,delimiter=',') #save the label
    numpy.savetxt(newy[1],numpy.asarray(prob_matrix),delimiter=',')
    print "New label has been generated!"

    #weights,bias=sda.sda_show_weights()

    #print weights[1],bias[1]



def main(argv=sys.argv):

    #debug-train:
    #sys.argv = [sys.argv[0], 'DeepLearning-Train', 'm_train.csv', '0.9', '11', '0.1', '0.1', '2', '2', '1', '[100]*2', '[0.1]*2']

    #debug-extend:
    #sys.argv = [sys.argv[0], 'DeepLearning-Extend', 'm_train.csv', 'm_extend.csv', '11', '[100]*2']


    if sys.argv[1]=="DeepLearning-Train":

        # this is separate use of deep learning

        #######################################
        ####### calling R  preprocessing   ####
        #######################################

        code_dir=os.getcwd()

        dpdata_address=os.path.split(os.path.dirname(sys.argv[2]))[-1]

        r=ro.r

        wd="".join(['setwd(\'',os.path.join(os.path.split(os.getcwd())[0],dpdata_address),'\')']).replace("\\","/")

        r(wd) #这句会改变python的当前路径

        r.source(os.path.join(code_dir,'DeepLearning-PreProcessing.R'))

        ro.globalenv['train_test_ratio']=float(sys.argv[3])

        ro.globalenv['training_data']=sys.argv[2]

        ro.globalenv['label_no']=int(sys.argv[4])

        inout=r('train_preprocessing(train_test_ratio,training_data,label_no)')

        no_input=int(inout[0])
        no_output=int(inout[1])

        #如果要自己定义一组特殊的start point就存放在这个folder里,hard negatvei mining就有这样的需求
        if sys.argv[12]=='random':
            new_initial_folder=None
        else:
            new_initial_folder=os.path.join(os.getcwd(),sys.argv[12])


        ########################################
        ##########

        data_header=os.path.split(sys.argv[2])[0]
        datasetname=[os.path.join(data_header,"DP_train.csv"),os.path.join(data_header,"DP_valid.csv"),
                     os.path.join(data_header,"DP_valid.csv")]

        training_learning_rate=float(sys.argv[5])

        tuning_learning_rate=float(sys.argv[6])

        training_epochs=int(sys.argv[7])

        tuning_epochs=int(sys.argv[8])

        batchsize=int(sys.argv[9])

        hiddenlayers=eval(sys.argv[10])

        noise_level=eval(sys.argv[11])

        train_SdA(finetune_lr=tuning_learning_rate, pretraining_epochs=training_epochs,
                pretrain_lr=training_learning_rate, training_epochs=tuning_epochs,
                dataset=datasetname,
                n_in=no_input,n_out=no_output,
                batch_size=batchsize, hidden_layers=hiddenlayers,
                corruption_levels=noise_level,
                weights_file=os.path.join(os.getcwd(),sys.argv[13]), #where to save the weights
                bias_file=os.path.join(os.getcwd(),sys.argv[13]),   #如果有指定的初始权值，那么就要换成新的名字
                weights_initial=new_initial_folder,bias_initial=new_initial_folder)  #如果有特定的初始权值，就从这里读入


    elif sys.argv[1]=='DeepLearning-Extend':
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

        filehead=os.path.join(os.getcwd(),sys.argv[6])


        data_header=os.path.split(sys.argv[2])[0]

        resulthead=[os.path.join(data_header,"DP_result.csv"),os.path.join(data_header,"DP_result_all.csv")]
        #results name，一个存label,一个存概率矩阵


        extend_SdA(newx=os.path.join(data_header,"DP_extend.csv"),
                            newy=resulthead,
                            weights_file=filehead,
                            bias_file=filehead,
                            n_in=no_input,n_out=no_output,hidden_layers=hiddenlayers)


        #########################################
        ### post-precessing the results using R #
        #########################################

        # r('rm(list=ls())')
        #
        # r(wd)
        #
        # ro.globalenv['extending_data']=sys.argv[3]
        #
        # ro.globalenv['non_feature']=int(sys.argv[4])
        #
        # ro.globalenv['num_clusters']=float(sys.argv[5])
        #
        # r.source('../dpdata/multiple_optimize.R')
        #
        # r('postprocessing(num_clusters,extending_data,non_feature)')



    else:

        raise ValueError("Please Input \'DeepLearning-Extend\' or \'DeepLearning-Train\' ")






if __name__ == '__main__':


    main()


   #python SdA.py DeepLearning-Train ../dpdata/m_train.csv 0.9 11 0.1 0.1 2 2 1 [100]*2 [0.1]*2 random 1_weights

   #python SdA.py DeepLearning-Train ../dpdata/m_train.csv 0.9 11 0.1 0.1 2 2 1 [100]*2 [0.1]*2 1_weights 2_weights

   #python SdA.py DeepLearning-Extend ../dpdata/m_train.csv m_extend.csv 11 [100]*2 1_weights

   #python SdA.py DeepLearning-Extend ../dpdata/m_train.csv m_extend.csv 11 [100]*2 2_weights