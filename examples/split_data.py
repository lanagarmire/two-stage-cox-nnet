from cox_nnet import *

import numpy

import sklearn

x = numpy.loadtxt(fname="/nfs/turbo/umms-garmire/jingzhe/two-stage-cox-nnet/data/coxnet_x.csv",delimiter=",",skiprows=0)

ytime = numpy.loadtxt(fname="/nfs/turbo/umms-garmire/jingzhe/two-stage-cox-nnet/data/y_days.csv",delimiter=",",skiprows=0)

ystatus = numpy.loadtxt(fname="/nfs/turbo/umms-garmire/jingzhe/two-stage-cox-nnet/data/event.csv",delimiter=",",skiprows=0)

count = 0
for i in range(20):
    count += 1
    x_train, x_test, ytime_train, ytime_test, ystatus_train, ystatus_test = sklearn.cross_validation.train_test_split(x, ytime, ystatus, test_size = 0.2, random_state = count)
    
    numpy.savetxt("/nfs/turbo/umms-garmire/jingzhe/split_data/x_train"+str(count)+".csv",x_train, delimiter=",")
    numpy.savetxt("/nfs/turbo/umms-garmire/jingzhe/split_data/x_test"+str(count)+".csv",x_test, delimiter=",")
    numpy.savetxt("/nfs/turbo/umms-garmire/jingzhe/split_data/ytime_train"+str(count)+".csv",ytime_train, delimiter=",")
    numpy.savetxt("/nfs/turbo/umms-garmire/jingzhe/split_data/ytime_test"+str(count)+".csv",ytime_test, delimiter=",")
    numpy.savetxt("/nfs/turbo/umms-garmire/jingzhe/split_data/ystatus_train"+str(count)+".csv",ystatus_train, delimiter=",")
    numpy.savetxt("/nfs/turbo/umms-garmire/jingzhe/split_data/ystatus_test"+str(count)+".csv",ystatus_test, delimiter=",")
