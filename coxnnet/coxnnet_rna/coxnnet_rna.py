from cox_nnet import *

import numpy

import sklearn
x = numpy.loadtxt(fname="./../../data/coxnet_x_rna.csv",delimiter=",",skiprows=0)

ytime = numpy.loadtxt(fname="./../../data/y_days_rna.csv",delimiter=",",skiprows=0)

ystatus = numpy.loadtxt(fname="./../../data/event_rna.csv",delimiter=",",skiprows=0)


# split into test/train sets

#x_train, x_test, ytime_train, ytime_test, ystatus_train, ystatus_test = sklearn.cross_validation.train_test_split(x, ytime, ystatus, test_size = 0.2)
#kf = cross_validation.KFold(313, n_folds=5, random_state=2019)
count = 0
for i in range(10):
    count += 1
 #   print("TRAIN:", train_index, "TEST:", test_index)
    #x_train, x_test = x[train_index], x[test_index]
    #ytime_train, ytime_test = ytime[train_index], ytime[test_index]
    #ystatus_train, ystatus_test = ystatus[train_index], ystatus[test_index]
    x_train, x_test, ytime_train, ytime_test, ystatus_train, ystatus_test = sklearn.cross_validation.train_test_split(x, ytime, ystatus, test_size = 0.2, random_state = count)

    model_params = dict(node_map = None, input_split = None)

    search_params = dict(method = "nesterov", learning_rate=0.005, momentum=0.9,

       max_iter=2000, stop_threshold=0.995, patience=1000, patience_incr=2, rand_seed = 100,

       eval_step=23, lr_decay = 0.9, lr_growth = 1.0)

    #search_params = dict(method = "adam", learning_rate=0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, max_iter=5000, stop_threshold=0.995, patience=1000, patience_incr=2,
    #rand_seed = 123, eval_step=23, lr_decay = 0.9, lr_growth = 1.0)	
    
    cv_params = dict(cv_seed=1, n_folds=5, cv_metric = "loglikelihood", L2_range = numpy.arange(-4,3,0.5))

    cv_likelihoods, L2_reg_params, mean_cvpl = L2CVProfile(x_train,ytime_train,ystatus_train,

        model_params,search_params,cv_params, verbose=False)
    L2_reg = L2_reg_params[numpy.argmax(mean_cvpl)]

    model_params = dict(node_map = None, input_split = None, L2_reg=numpy.exp(L2_reg))

    model, cost_iter = trainCoxMlp(x_train, ytime_train, ystatus_train, model_params, search_params, verbose=True)
    theta = model.predictNewData(x_train)
    theta_2 = model.predictNewData(x_test)
    
    #featureScore = varImportance(model, x_train, ytime_train, ystatus_train)

    #numpy.savetxt('./../output/rna_feature_importance'+str(count)+'.csv', featureScore, delimiter=",")
    
    numpy.savetxt('./../../output/rna_ypred_train_'+str(count)+'.csv', theta, delimiter=",")

    numpy.savetxt("./../../output/rna_ytime_train"+str(count)+'.csv', ytime_train, delimiter=",")

    numpy.savetxt("./../../output/rna_ystatus_train"+str(count)+'.csv', ystatus_train, delimiter=",")
    
    numpy.savetxt('./../../output/rna_ypred_test_'+str(count)+'.csv', theta_2, delimiter=",")

    numpy.savetxt("./../../output/rna_ytime_test"+str(count)+'.csv', ytime_test, delimiter=",")

    numpy.savetxt("./../../output/rna_ystatus_test"+str(count)+'.csv', ystatus_test, delimiter=",")






