#-*- coding:utf-8 -*-

import os
import sys
import time
import csv
import numpy
import rpy2.robjects as ro
import theano
import theano.tensor as T
from theano import pp
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
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 17))

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

                weights_filename=[hiddenLayer_weights_file,str(i+1),"_hiddenLayer_W.csv"]
                bias_filename=[hiddenLayer_bias_file,str(i+1),"_hiddenLayer_b.csv"]

                f=open("".join(weights_filename),"rb")
                data=numpy.loadtxt(f,delimiter=',',dtype=float)
                f.close()
                shared_hiddenLayer_W = theano.shared(numpy.asarray(data,dtype=theano.config.floatX),borrow=True)

                f=open("".join(bias_filename),"rb")
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

        #最后加一层logistic
        if logLayer_weights_file is None:

            self.logLayer = LogisticRegression(
                input=self.sigmoid_layers[-1].output,
                n_in=hidden_layers_sizes[-1],
                n_out=n_outs
            )

        else:

            print "Reading in the weights and bias of Logistic Layer \n"

            #把logistic layer的权重读入
            weights_filename=[logLayer_weights_file,"logisticLayer_W.csv"]
            bias_filename=[logLayer_bias_file,"logisticLayer_b.csv"]

            f=open("".join(weights_filename),"rb")
            data=numpy.loadtxt(f,delimiter=',',dtype=float)
            f.close()
            shared_logLayer_W = theano.shared(numpy.asarray(data,dtype=theano.config.floatX),borrow=True)

            f=open("".join(bias_filename),"rb")
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
                    self.x: train_set_x[batch_begin: batch_end]
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

        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, valid_score, test_score

    def sda_show_weights(self):

        weights=[]

        bias=[]

        # the weights of hidden layer
        # use the dictionary to save
        for i in xrange(self.n_layers):

            #weights["Hidden"+str(i+1)]=self.sigmoid_layers[i].W.get_value()
            #bias["Hidden"+str(i+1)]=self.sigmoid_layers[i].b.get_value()
            weights.append(self.sigmoid_layers[i].W.get_value())
            bias.append(self.sigmoid_layers[i].b.get_value())



        #weights["Final"]=self.logLayer.W.get_value()
        #bias["Final"]=self.logLayer.b.get_value()
        weights.append(self.logLayer.W.get_value())
        bias.append(self.logLayer.b.get_value())

        return(weights,bias)


    def sda_show_pretraining_labels(self, extendinput,i):

        if i==0:
            return T.nnet.sigmoid(T.dot(extendinput, self.sigmoid_layers[i].W) + self.sigmoid_layers[i].b)
        else:
            return T.nnet.sigmoid(T.dot(self.sda_show_pretraining_labels(extendinput,i-1),
                                        self.sigmoid_layers[i].W) + self.sigmoid_layers[i].b)

    def sda_show_labels(self,extendinput):

        self.p_newy_given_x = T.nnet.softmax(T.dot(self.sda_show_pretraining_labels(extendinput,self.n_layers-1)
                                                   , self.logLayer.W) + self.logLayer.b)

        self.p_newy = T.cast(T.argmax(self.p_newy_given_x, axis=1),'int32')

        return (self.p_newy,T.max(self.p_newy_given_x,axis=1))











def test_SdA(finetune_lr=0.01, pretraining_epochs=30,
             pretrain_lr=0.01, training_epochs=1000,
             dataset=[], n_in=29,n_out=2,
             batch_size=100, hidden_layers=[],
             corruption_levels=[],
             newx=None,newy=None,weights_file=None,bias_file=None):

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
        n_outs=n_out
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
                         corruption=corruption_levels[i],
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
        datasets=datasets,
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

            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model()
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

        weights,bias=sda.sda_show_weights()

        if len(weights)<>len(hidden_layers)+1:

            raise ValueError("The number of hidden layers is wrong!")


        #记录logistic layer的权重
        weights_filename=[weights_file,"logisticLayer_W.csv"]
        bias_filename=[bias_file,"logisticLayer_b.csv"]
        numpy.savetxt("".join(weights_filename),weights.pop(),delimiter=",")
        numpy.savetxt("".join(bias_filename) ,bias.pop(),delimiter=",")

        #记录hidden layer的权重
        for i in xrange(len(weights)):
            weights_filename=[weights_file,str(i+1),"_hiddenLayer_W.csv"]
            bias_filename=[bias_file,str(i+1),"_hiddenLayer_b.csv"]
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



def SdA_extend_as_you_want(newx=None,newy=None,
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
    print "New label has been generated!"




def main(argv=sys.argv):


    if sys.argv[1]=="LCG-CC-Train":


        #############################
        ### pre-processing using R ##
        #############################

        print ('\nCalling R to pre-processing the data\n')

        r=ro.r

        wd="".join(['setwd(\'' ,os.getcwd().replace('code','dpdata'),'\')'])

        r(wd)

        r.source('../dpdata/LCG-CC-PreProcessing.R')

        ro.globalenv['pl_threshold']=float(sys.argv[13])

        ro.globalenv['num_clusters']=int(sys.argv[12])

        ro.globalenv['train_test_ratio']=float(sys.argv[3])

        ro.globalenv['training_data']=sys.argv[2]

        ro.globalenv['label_no']=int(sys.argv[4])

        no_input=int(r('train_preprocessing(pl_threshold,num_clusters,train_test_ratio,training_data,label_no)')[0])

        #preprocessing return the number of output


        no_output=2

        #############################
        ### Training the model   ####
        #############################

        training_learning_rate=float(sys.argv[5])

        tuning_learning_rate=float(sys.argv[6])

        training_epochs=int(sys.argv[7])

        tuning_epochs=int(sys.argv[8])

        batchsize=int(sys.argv[9])

        hiddenlayers=eval(sys.argv[10])

        noise_level=eval(sys.argv[11])




        workenv="../dpdata/classifier"    #存权重

        filehead="../dpdata/multiple_dl_" #存结果

        noofclu=[x+1 for x in range(int(sys.argv[12]))]

        # noofclu=[6]

        datasetname=["_train.csv","_valid.csv","_valid.csv"] #这里的3个文件是传统的算法理的，必须都有Label
                                                        #train,valid是正常的，test没有Label,test_pl有label有Pl

        for i in noofclu:

            print "Building Sub-Classifier:  %d" %i
            dataset=[]

            for j in datasetname:
                dataset_name=[filehead,str(i),j]
                dataset.append("".join(dataset_name))

            #print dataset
            resultname=[filehead,str(i),"_result_on_valid.csv"]

            test_SdA(finetune_lr=tuning_learning_rate, pretraining_epochs=training_epochs,
                    pretrain_lr=training_learning_rate, training_epochs=tuning_epochs,
                    dataset=dataset,
                    n_in=no_input,n_out=no_output,
                    batch_size=batchsize, hidden_layers=hiddenlayers,
                    corruption_levels=noise_level,
                    newx="".join([filehead,str(i),"_test.csv"]),newy="".join(resultname),
                    weights_file="".join([workenv,'_',str(i),'/']),
                    bias_file="".join([workenv,'_',str(i),'/']) )


    elif sys.argv[1]=="LCG-CC-Extend":

        #####################################
        ### Extending the model on new data #
        #####################################

        #注意extend的时候要看清楚当前读入的weights是哪个pl定义下的,weights的clasifier文件名前面没有pl的名字


        noofclu=[x+1 for x in range(int(sys.argv[6]))]

        ro.globalenv['training_data']=sys.argv[2]

        ro.globalenv['extending_data']=sys.argv[3]

        ro.globalenv['non_feature']=int(sys.argv[4])


        hiddenlayers=eval(sys.argv[5])


        print ('\nCalling R to pre-processing the data\n')

        r=ro.r

        wd="".join(['setwd(\'' ,os.getcwd().replace('code','dpdata'),'\')'])

        r(wd)

        r.source('../dpdata/LCG-CC-PreProcessing.R')

        #normalize the data using R
        no_input=int(r('extend_preprocessing(training_data,extending_data, non_feature)')[0])

        no_output=2



        filehead="../dpdata/classifier_"

        resulthead='../dpdata/multiple_dl_'


        for i in noofclu:

            print "In Sub-Classifier:  %d" %i

            SdA_extend_as_you_want(newx="../dpdata/multiple_extend_set.csv",
                                newy="".join([resulthead,str(i),"_extend.csv"]),
                                weights_file="".join([filehead,str(i),'/']),
                                bias_file="".join([filehead,str(i),'/']),
                                n_in=no_input,n_out=no_output,hidden_layers=hiddenlayers)


        #########################################
        ### post-precessing the results using R #
        #########################################

        r('rm(list=ls())')

        r(wd)

        ro.globalenv['extending_data']=sys.argv[3]

        ro.globalenv['non_feature']=int(sys.argv[4])

        ro.globalenv['num_clusters']=float(sys.argv[6])

        r.source('../dpdata/LCG-CC-PostProcessing.R')

        r('postprocessing(num_clusters,extending_data,non_feature)')





    elif sys.argv[1]=="DeepLearning-Train":

        # this is separate use of deep learning

        #######################################
        ####### calling R  preprocessing   ####
        #######################################

        r=ro.r

        wd="".join(['setwd(\'' ,os.getcwd().replace('code','dpdata'),'\')'])

        r(wd)

        r.source('../dpdata/DeepLearning-PreProcessing.R')


        ro.globalenv['train_test_ratio']=float(sys.argv[3])

        ro.globalenv['training_data']=sys.argv[2]

        ro.globalenv['label_no']=int(sys.argv[4])

        inout=r('train_preprocessing(train_test_ratio,training_data,label_no)')

        no_input=int(inout[0])
        no_output=int(inout[1])
        ########################################
        ##########

        datasetname=["../dpdata/DP_train.csv","../dpdata/DP_valid.csv","../dpdata/DP_valid.csv"]

        training_learning_rate=float(sys.argv[5])

        tuning_learning_rate=float(sys.argv[6])

        training_epochs=int(sys.argv[7])

        tuning_epochs=int(sys.argv[8])

        batchsize=int(sys.argv[9])

        hiddenlayers=eval(sys.argv[10])

        noise_level=eval(sys.argv[11])

        test_SdA(finetune_lr=tuning_learning_rate, pretraining_epochs=training_epochs,
                pretrain_lr=training_learning_rate, training_epochs=tuning_epochs,
                dataset=datasetname,
                n_in=no_input,n_out=no_output,
                batch_size=batchsize, hidden_layers=hiddenlayers,
                corruption_levels=noise_level,
                #newx="".join([filehead,str(i),"_test.csv"]),newy="".join(resultname),
                weights_file="../dpdata/DP_classifier/" ,
                bias_file="../dpdata/DP_classifier/"  )

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

        r=ro.r

        wd="".join(['setwd(\'' ,os.getcwd().replace('code','dpdata'),'\')'])

        r(wd)

        r.source('../dpdata/DeepLearning-PreProcessing.R')

        #normalize the data using R
        inout=r('extend_preprocessing(training_data,extending_data, non_feature)')

        no_input=int(inout[0])
        no_output=int(inout[1])

        filehead="../dpdata/DP_classifier/"               # weights

        resulthead='../dpdata/DP_result.csv' #results name


        SdA_extend_as_you_want(newx="../dpdata/DP_extend.csv",
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

        raise ValueError("Please Input \'LCG-CC-Train\' or \'LCG-CC-Extend\' \n \'DeepLearning-Extend\' or \'DeepLearning-Train\' ")






if __name__ == '__main__':

   main()

   #python SdA.py LCG-CC-Extend multiple_deeplearning.csv multiple_deeplearning_extend.csv 19 [100]*2 5

   #python SdA.py LCG-CC-Train multiple_deeplearning.csv 0.8 19 0.1 0.1 2 2 1 [100]*2 [0.3]*2 5 1100

   #python SdA.py DeepLearning-Train m_train.csv 0.9 11 0.1 0.1 2 2 1 [100]*2 [0.1]*2

   #python SdA.py DeepLearning-Extend m_train.csv m_extend.csv 11 [100]*2