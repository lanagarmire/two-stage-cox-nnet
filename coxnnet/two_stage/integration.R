library(tidyverse)  # data manipulation
library(caret) 
library(glmnet)
library(survival)
library(ggplot2)

for (k in 1:10){

   ###########Image Hidden Layer
   W = read.csv(paste(paste('./../../output/W',k,sep=''),'csv',sep='.'),header =F)
   b= read.csv(paste(paste('./../../output/b',k,sep=''),'csv',sep='.'),header= F)
   W = t(W)
   b = t(b)
   test = read.csv(paste(paste('./../../output/coxnet_imagetest',k,sep=''),'csv',sep='.'), header = F)
   train = read.csv(paste(paste('./../../output/coxnet_imagetrain',k,sep=''),'csv',sep='.'), header = F)
   
   test= scale(test)
   test[is.na(test)] = 0
   
   train =scale(train)
   train[is.na(train)] = 0

   image_hidden_test = matrix(rep(0,50*58),nrow = 58) # 58 is the sample size ; 50 is sqrt(number of features)   
   for (i in 1:58){
     for (j in 1:50){
       image_hidden_test[i,j] = tanh(sum(test[i,]*W[j,])+b[j])
     }
   }
   
   image_hidden_train = matrix(rep(0,50*232),nrow = 232)
   for (i in 1:232){
      for (j in 1:50){
         image_hidden_train[i,j] = tanh(sum(train[i,]*W[j,])+b[j])  
      }
   }
   
   time_test = read.csv(paste(paste('./../../output/coxnnet_imagetest_time',k,sep=''),'csv',sep='.'),header=F)
   sta_test = read.csv(paste(paste('./../../output/coxnnet_imagetest_sta',k,sep=''),'csv',sep='.'),header=F)
   Ymatrix_test = Surv(time_test[,1],sta_test[,1])
   
   time_train = read.csv(paste(paste('./../../output/coxnnet_imagetrain_time',k,sep=''),'csv',sep='.'),header=F)
   sta_train = read.csv(paste(paste('./../../output/coxnnet_imagetrain_sta',k,sep=''),'csv',sep='.'),header=F)
   Ymatrix_train = Surv(time_train[,1],sta_train[,1])
   
   cv.tr=cv.glmnet(as.matrix(image_hidden_train),Ymatrix_train,family='cox',alpha=0.9,nfolds=10)
   
   predTrain=predict(cv.tr,as.matrix(image_hidden_train),s=cv.tr$lambda.min,type='response')
   predTest<-predict(cv.tr,as.matrix(image_hidden_test),s=cv.tr$lambda.min,type='response')
   
   survConcordance(Ymatrix_train ~ predTrain)$concordance
   survConcordance(Ymatrix_test ~ predTest)$concordance



   ############RNA Hidden Layer
   
   r = read.csv(paste(paste('./../../output/coxnet_rtrain',k,sep=''),'csv',sep='.'), header = F)
   W_r = read.csv(paste(paste('./../../output/Wr',k,sep=''),'csv',sep='.'),header = F)
   b_r = read.csv(paste(paste('./../../output/br',k,sep=''),'csv',sep='.'),header = F)
   W_r = t(W_r)
   b_r = t(b_r)
   
   rte = read.csv(paste(paste('./../../output/coxnet_rtest',k,sep=''),'csv',sep='.'),header = F)
   rte = scale(rte)
   rte[is.na(rte)] = 0
   
   Gmatrix = r
   Gmatrix = scale(Gmatrix)
   Gmatrix[is.na(Gmatrix)] = 0
   
   rna_hidden_tr = matrix(rep(0,126*232),nrow = 232)
   rna_hidden_te = matrix(rep(0,126*58),nrow = 58)
   
   for (i in 1:232){ # 232 is the sample siz of training
      for (j in 1:126){ # 126 is the sqrt(number of gene features)
         rna_hidden_tr[i,j] = tanh(sum(Gmatrix[i,]*W_r[j,])+b_r[j])  
      }
   }

   for (i in 1:58){
      for (j in 1:126){
         rna_hidden_te[i,j] = tanh(sum(rte[i,]*W_r[j,])+b_r[j])
      }
   }
   
   # integration of hidden features
   inte_feature_tr = cbind(image_hidden_train,rna_hidden_tr)
   inte_feature_tr = scale(inte_feature_tr)
   write.csv(inte_feature_tr, paste(paste('./../../output/rna_image_tr',k,sep=''),'csv',sep='.'),row.names = F,col.names = F)
   inte_feature_te = cbind(image_hidden_test,rna_hidden_te)
   inte_feature_te = scale(inte_feature_te)
   write.csv(inte_feature_te, paste(paste('./../../output/rna_image_te',k,sep=''),'csv',sep='.'),row.names = F,col.names = F)
   write.csv(time_test,paste(paste('./../../output/time_test',k,sep=''),'csv',sep='.'),row.names=F,col.names = F)
   write.csv(time_train,paste(paste('./../../output/time_train',k,sep=''),'csv',sep='.'),row.names=F,col.names = F)
   write.csv(sta_test,paste(paste('./../../output/sta_test',k,sep=''),'csv',sep='.'),row.names = F,col.names = F)
   write.csv(sta_train,paste(paste('./../../output/sta_train',k,sep=''),'csv',sep='.'),row.names=F,col.names = F)
}


#rna_hidden_te = scale(rna_hidden_te)
#rna_hidden_tr = scale(rna_hidden_tr)


# coxph elastic net, on rna hidden features
#cv.tr=cv.glmnet(as.matrix(rna_hidden_tr),Ymatrix_train,family='cox',alpha=0.9,nfolds=10)

#predTrain=predict(cv.tr,as.matrix(rna_hidden_tr),s=cv.tr$lambda.min,type='response')
#predTest<-predict(cv.tr,as.matrix(rna_hidden_te),s=cv.tr$lambda.min,type='response')

#survConcordance(Ymatrix_train ~ predTrain)$concordance
#survConcordance(Ymatrix_test ~ predTest)$concordance

# coxph elastic net, on integrated rna+image hidden features
#cv.tr=cv.glmnet(as.matrix(inte_feature_tr),Ymatrix_train,family='cox',alpha=0.9,nfolds=10)

#predTrain=predict(cv.tr,as.matrix(inte_feature_tr),s=cv.tr$lambda.min,type='response')
#predTest<-predict(cv.tr,as.matrix(inte_feature_te),s=cv.tr$lambda.min,type='response')

#survConcordance(Ymatrix_train ~ predTrain)$concordance
#survConcordance(Ymatrix_test ~ predTest)$concordance


# Rsquare: regression of 50 image-hidden features on all rna-hidden features
#rSq = c()
#image_hidden = rbind(image_hidden_train,image_hidden_test)
#rna_hidden = rbind(rna_hidden_tr,rna_hidden_te)
#
#for (i in 1:50){
#  y = image_hidden[,i]
#  model = cv.glmnet(as.matrix(rna_hidden),y,family="gaussian",alpha = 1, nfolds = 5) # cox regression with lasso penalty
#  cf = coef(model, s = 'lambda.min')
#  cc2 = as.matrix(cf[-1,])
#  predict = rna_hidden %*% cc2 
#  err = predict - y
#  r2 = 1-var(err)/var(y)
#  rSq = c(rSq,r2)
#}
#rSq


