from cox_nnet import *

import numpy

import sklearn

count = 0
for i in range(5):
    count += 1
    x_train = numpy.loadtxt(fname="rna_image_tr"+str(count)+".csv",delimiter=",",skiprows=1)
    
    ytime_train = numpy.loadtxt(fname="time_train"+str(count)+".csv",delimiter=",",skiprows=1)
    
    ystatus_train = numpy.loadtxt(fname="sta_train"+str(count)+".csv",delimiter=",",skiprows=1)

    x_test = numpy.loadtxt(fname="rna_image_te"+str(count)+".csv",delimiter=",",skiprows=1)

    ytime_test = numpy.loadtxt(fname="time_test"+str(count)+".csv",delimiter=",",skiprows=1)

    ystatus_test = numpy.loadtxt(fname="sta_test"+str(count)+".csv",delimiter=",",skiprows=1)

    model_params = dict(node_map = None, input_split = None)
    
    search_params = dict(method = "nesterov", learning_rate=0.005, momentum=0.9,
    
        max_iter=2000, stop_threshold=0.995, patience=1000, patience_incr=2, rand_seed = 100,
    
        eval_step=23, lr_decay = 0.9, lr_growth = 1.0)
    
    cv_params = dict(cv_seed=1, n_folds=5, cv_metric = "loglikelihood", L2_range = numpy.arange(-4,3,0.5))
    
    cv_likelihoods, L2_reg_params, mean_cvpl = L2CVProfile(x_train,ytime_train,ystatus_train,
    
        model_params,search_params,cv_params, verbose=False)
    L2_reg = L2_reg_params[numpy.argmax(mean_cvpl)]
    
    model_params = dict(node_map = None, input_split = None, L2_reg=numpy.exp(L2_reg))
    
    model, cost_iter = trainCoxMlp(x_train, ytime_train, ystatus_train, model_params, search_params, verbose=True)
    theta = model.predictNewData(x_test)
    theta_2 = model.predictNewData(x_train)
    
    numpy.savetxt('iR_ypred_test'+str(count)+'.csv', theta, delimiter=",")
    
    numpy.savetxt('iR_ytime_test'+str(count)+'.csv', ytime_test, delimiter=",")
    
    numpy.savetxt('iR_ystatus_test'+str(count)+'.csv', ystatus_test, delimiter=",")
    numpy.savetxt('iR_ypred_train'+str(count)+'.csv', theta_2, delimiter=",")
    
    numpy.savetxt('iR_ytime_train'+str(count)+'.csv', ytime_train, delimiter=",")
    
    numpy.savetxt('iR_ystatus_train'+str(count)+'.csv', ystatus_train, delimiter=",")
