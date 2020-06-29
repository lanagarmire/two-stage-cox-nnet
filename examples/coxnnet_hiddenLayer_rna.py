from cox_nnet import *

import numpy

import sklearn
x = numpy.loadtxt(fname="coxnet_x_rna.csv",delimiter=",",skiprows=0)

ytime = numpy.loadtxt(fname="y_days_rna.csv",delimiter=",",skiprows=0)

ystatus = numpy.loadtxt(fname="event_rna.csv",delimiter=",",skiprows=0)

count = 0
for i in range(5):
    count += 1
    x_train, x_test, ytime_train, ytime_test, ystatus_train, ystatus_test = sklearn.cross_validation.train_test_split(x, ytime, ystatus, test_size = 0.2, random_state = count)

    # remove sample TCGA-2Y-A9GZ-01
    #x_train = numpy.delete(x_train, 35, axis=0)
    #ytime_train = numpy.delete(ytime_train, 35)
    #ystatus_train = numpy.delete(ystatus_train, 35)
    
    
    model_params = dict(node_map = None, input_split = None)
    
    search_params = dict(method = "nesterov", learning_rate=0.01, momentum=0.9,
        max_iter=2000, stop_threshold=0.995, patience=1000, patience_incr=2, rand_seed = 123,

        eval_step=23, lr_decay = 0.9, lr_growth = 1.0)

    cv_params = dict(cv_seed=1, n_folds=5, cv_metric = "loglikelihood", L2_range = numpy.arange(-4.5,1,0.5))
    
    
    #cross validate training set to determine lambda parameters
    
    cv_likelihoods, L2_reg_params, mean_cvpl = L2CVProfile(x,ytime,ystatus,
    
        model_params,search_params,cv_params, verbose=False)
    

    
    #build model based on optimal lambda parameter
    
    L2_reg = L2_reg_params[numpy.argmax(mean_cvpl)]
    
    model_params = dict(node_map = None, input_split = None, L2_reg=numpy.exp(L2_reg))
    
    model, cost_iter = trainCoxMlp(x_train, ytime_train, ystatus_train, model_params, search_params, verbose=True)
    
    
    
    b = map(lambda tvar : tvar.eval(), model.b)
    W = map(lambda tvar : tvar.eval(), model.W)
    b=b[0]
    W=W[0]
    
    # use built-in functions for integration
    ## b=numpy.asarray(b,dtype='float32')
    ## W=numpy.asarray(W,dtype='float32')
    ## rna_hidden = hidden_features(W,b,x_test)
    ## numpy.savetxt("rna_hidden1.csv",rna_hidden,delimiter=",")

    numpy.savetxt("Wr"+str(count)+".csv", W, delimiter=",")
    numpy.savetxt("br"+str(count)+".csv", b, delimiter=",")
    numpy.savetxt("coxnet_rtest"+str(count)+".csv",x_test, delimiter=",")
    numpy.savetxt("coxnnet_rtest_time"+str(count)+".csv",ytime_test,delimiter = ",")
    numpy.savetxt("coxnnet_rtest_sta"+str(count)+".csv", ystatus_test, delimiter = ",")
    numpy.savetxt("coxnet_rtrain"+str(count)+".csv",x_train, delimiter=",")
    numpy.savetxt("coxnnet_rtrain_time"+str(count)+".csv",ytime_train,delimiter = ",")
    numpy.savetxt("coxnnet_rtrain_sta"+str(count)+".csv", ystatus_train, delimiter = ",")


