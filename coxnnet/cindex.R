library(tidyverse)  # data manipulation
library(caret)
library(glmnet)
library(survival)
library(ggplot2)

test_cindex_coxph_gene = c()
train_cindex_coxph_gene = c()
test_cindex_coxph_image = c()
train_cindex_coxph_image = c()
test_cindex_image = c()
train_cindex_image = c()
test_cindex_gene = c()
train_cindex_gene = c()
test_cindex_ir = c()
train_cindex_ir = c()

SEED=5
k=10
for (i in 1:k){

   # coxph (gene)
   test = read.csv(paste(paste('./../output/coxnet_rtest',i,sep=''),'csv',sep='.'), header = F)
   train = read.csv(paste(paste('./../output/coxnet_rtrain',i,sep=''),'csv',sep='.'), header = F)

   test= scale(test)
   test[is.na(test)] = 0

   train =scale(train)
   train[is.na(train)] = 0

   time_test = read.csv(paste(paste("./../output/coxnnet_rtest_time",i,sep=''),'csv',sep='.'),header=F)
   sta_test = read.csv(paste(paste("./../output/coxnnet_rtest_sta",i,sep=''),'csv',sep='.'),header=F)

   time_train = read.csv(paste(paste("./../output/coxnnet_rtrain_time",i,sep=''),'csv',sep='.'),header=F)
   sta_train = read.csv(paste(paste("./../output/coxnnet_rtrain_sta",i,sep=''),'csv',sep='.'),header=F)


   Ymatrix_test = Surv(time_test[,1],sta_test[,1])
   Ymatrix_train = Surv(time_train[,1],sta_train[,1])

   set.seed(12345)
   cv.tr=cv.glmnet(as.matrix(train),Ymatrix_train,family='cox',alpha=0.9,nfolds=10)

   predTest<-predict(cv.tr,as.matrix(test),s=cv.tr$lambda.min,type='response')
   predTest = predTest[,1]
   predTrain=predict(cv.tr,as.matrix(train),s=cv.tr$lambda.min,type='response')
   predTrain = predTrain[,1]


   test_km = cbind(predTest, time_test[,1], sta_test[,1])
   train_km = cbind(predTrain, time_train[,1], sta_train[,1])
   colnames(test_km) = c("risk","time","status")
   colnames(train_km) = c("risk","time","status")
   
   if(i==SEED){
      write.csv(test_km, paste("KM_plots", "/coxph_gene_test.csv",sep=''), row.names=FALSE)
      write.csv(train_km, paste("KM_plots", "/coxph_gene_train.csv",sep=''), row.names=FALSE)
   }

   cindex_train = survConcordance(Ymatrix_train ~ predTrain)$concordance
   cindex_test =survConcordance(Ymatrix_test ~ predTest)$concordance

   test_cindex_coxph_gene = c(test_cindex_coxph_gene,cindex_test)
   train_cindex_coxph_gene =c(train_cindex_coxph_gene,cindex_train)


   # coxph (image)
   test = read.csv(paste(paste('./../output/coxnet_imagetest',i,sep=''),'csv',sep='.'), header = F)
   train = read.csv(paste(paste('./../output/coxnet_imagetrain',i,sep=''),'csv',sep='.'), header = F)
   
   test= scale(test)
   test[is.na(test)] = 0
   
   train =scale(train)
   train[is.na(train)] = 0
   
   time_test = read.csv(paste(paste("./../output/coxnnet_imagetest_time",i,sep=''),'csv',sep='.'),header=F)
   sta_test = read.csv(paste(paste("./../output/coxnnet_imagetest_sta",i,sep=''),'csv',sep='.'),header=F)
   
   time_train = read.csv(paste(paste("./../output/coxnnet_imagetrain_time",i,sep=''),'csv',sep='.'),header=F)
   sta_train = read.csv(paste(paste("./../output/coxnnet_imagetrain_sta",i,sep=''),'csv',sep='.'),header=F)
  

   Ymatrix_test = Surv(time_test[,1],sta_test[,1])
   Ymatrix_train = Surv(time_train[,1],sta_train[,1])
   
   set.seed(12345)
   cv.tr=cv.glmnet(as.matrix(train),Ymatrix_train,family='cox',alpha=0.9,nfolds=10)
   
   predTest<-predict(cv.tr,as.matrix(test),s=cv.tr$lambda.min,type='response')
   predTest = predTest[,1]
   predTrain=predict(cv.tr,as.matrix(train),s=cv.tr$lambda.min,type='response')
   predTrain = predTrain[,1]   

   
   test_km = cbind(predTest, time_test[,1], sta_test[,1])   
   train_km = cbind(predTrain, time_train[,1], sta_train[,1])   
   colnames(test_km) = c("risk","time","status")
   colnames(train_km) = c("risk","time","status")
   
   if(i==SEED){
      write.csv(test_km, paste("KM_plots", "/coxph_image_test.csv",sep=''), row.names=FALSE)
      write.csv(train_km, paste("KM_plots", "/coxph_image_train.csv",sep=''), row.names=FALSE)
   }

   cindex_train = survConcordance(Ymatrix_train ~ predTrain)$concordance
   cindex_test =survConcordance(Ymatrix_test ~ predTest)$concordance
   
   test_cindex_coxph_image = c(test_cindex_coxph_image,cindex_test)
   train_cindex_coxph_image =c(train_cindex_coxph_image,cindex_train)


   # coxnnet image
   time_test = read.csv(paste(paste("./../output/image_ytime_test",i,sep=''),"csv",sep='.'),header=F)
   sta_test = read.csv(paste(paste("./../output/image_ystatus_test",i,sep=''),"csv",sep='.'),header=F)
   Ymatrix_test = Surv(time_test[,1],sta_test[,1])

   time_train = read.csv(paste(paste("./../output/image_ytime_train",i,sep=''),"csv",sep='.'),header=F)
   sta_train = read.csv(paste(paste("./../output/image_ystatus_train",i,sep=''),"csv",sep='.'),header=F)
   Ymatrix_train = Surv(time_train[,1],sta_train[,1])

   predTest = read.csv(paste(paste("./../output/image_ypred_test",i,sep='_'),"csv",sep='.'),header=F)
   predTest = predTest[,1]
   predTrain = read.csv(paste(paste("./../output/image_ypred_train",i,sep='_'),"csv",sep='.'),header=F)
   predTrain = predTrain[,1]

   
   test_km = cbind(predTest, time_test[,1], sta_test[,1])
   train_km = cbind(predTrain, time_train[,1], sta_train[,1])
   colnames(test_km) = c("risk","time","status")
   colnames(train_km) = c("risk","time","status")
   
   if(i==SEED){
      write.csv(test_km, paste("KM_plots","/coxnet_image_test.csv",sep=''), row.names=FALSE)
      write.csv(train_km, paste("KM_plots","/coxnet_image_train.csv",sep=''), row.names=FALSE)
   }

   cindex_test = survConcordance(Ymatrix_test ~ predTest)$concordance
   cindex_train = survConcordance(Ymatrix_train ~ predTrain)$concordance

   test_cindex_image = c(test_cindex_image,cindex_test)
   train_cindex_image =c(train_cindex_image,cindex_train)
   

   # coxnnet gene
   time_test = read.csv(paste(paste("./../output/rna_ytime_test",i,sep=''),"csv",sep='.'),header=F)
   sta_test = read.csv(paste(paste("./../output/rna_ystatus_test",i,sep=''),"csv",sep='.'),header=F)
   Ymatrix_test = Surv(time_test[,1],sta_test[,1])

   time_train = read.csv(paste(paste("./../output/rna_ytime_train",i,sep=''),"csv",sep='.'),header=F)
   sta_train = read.csv(paste(paste("./../output/rna_ystatus_train",i,sep=''),"csv",sep='.'),header=F)
   Ymatrix_train = Surv(time_train[,1],sta_train[,1])

   predTest = read.csv(paste(paste("./../output/rna_ypred_test",i,sep='_'),"csv",sep='.'),header=F)
   predTest = predTest[,1]
   predTrain = read.csv(paste(paste("./../output/rna_ypred_train",i,sep='_'),"csv",sep='.'),header=F)
   predTrain = predTrain[,1]

   
   test_km = cbind(predTest, time_test[,1], sta_test[,1])
   train_km = cbind(predTrain, time_train[,1], sta_train[,1])
   colnames(test_km) = c("risk","time","status")
   colnames(train_km) = c("risk","time","status")
   
   if(i==SEED){
      write.csv(test_km, paste("KM_plots", "/coxnet_gene_test.csv",sep=''), row.names=FALSE)
      write.csv(train_km, paste("KM_plots", "/coxnet_gene_train.csv",sep=''), row.names=FALSE)
   }   

   cindex_test = survConcordance(Ymatrix_test ~ predTest)$concordance
   cindex_train = survConcordance(Ymatrix_train ~ predTrain)$concordance

   test_cindex_gene = c(test_cindex_gene,cindex_test)
   train_cindex_gene =c(train_cindex_gene,cindex_train)   


   # coxnnet image + gene
   time_test = read.csv(paste(paste("./../output/iR_ytime_test",i,sep=''),"csv",sep='.'),header=F)
   sta_test = read.csv(paste(paste("./../output/iR_ystatus_test",i,sep=''),"csv",sep='.'),header=F)
   Ymatrix_test = Surv(time_test[,1],sta_test[,1])
   
   time_train = read.csv(paste(paste("./../output/iR_ytime_train",i,sep=''),"csv",sep='.'),header=F)
   sta_train = read.csv(paste(paste("./../output/iR_ystatus_train",i,sep=''),"csv",sep='.'),header=F)
   Ymatrix_train = Surv(time_train[,1],sta_train[,1])
   
   predTest = read.csv(paste(paste("./../output/iR_ypred_test",i,sep=''),"csv",sep='.'),header=F)
   predTest = predTest[,1]
   predTrain = read.csv(paste(paste("./../output/iR_ypred_train",i,sep=''),"csv",sep='.'),header=F)
   predTrain = predTrain[,1]
   
   
   test_km = cbind(predTest, time_test[,1], sta_test[,1])
   train_km = cbind(predTrain, time_train[,1], sta_train[,1])
   colnames(test_km) = c("risk","time","status")
   colnames(train_km) = c("risk","time","status")
 
   if(i==SEED){
      write.csv(test_km, paste("KM_plots", "/coxnet_image_gene_test.csv",sep=''), row.names=FALSE)
      write.csv(train_km, paste("KM_plots", "/coxnet_image_gene_train.csv",sep=''), row.names=FALSE)
   }   

   cindex_test = survConcordance(Ymatrix_test ~ predTest)$concordance
   cindex_train = survConcordance(Ymatrix_train ~ predTrain)$concordance

   test_cindex_ir = c(test_cindex_ir,cindex_test)
   train_cindex_ir =c(train_cindex_ir,cindex_train)
}

cindex = c(test_cindex_coxph_gene, test_cindex_coxph_image, test_cindex_image, test_cindex_gene, test_cindex_ir, train_cindex_coxph_gene,train_cindex_coxph_image, train_cindex_image, train_cindex_gene, train_cindex_ir)
type = c(rep("test", k*5),rep("train", k*5))
model = c(rep("Cox-PH(gene)",k),rep("Cox-PH(image)",k),rep("Cox-nnet(image)",k),rep("Cox-nnet(Gene)",k),rep("Cox-nnet(image+Gene)",k))
model = c(model, model)
table = cbind(model, type, cindex)
colnames(table) = c("model","type","cindex")
write.csv(table, "./cindex.csv",row.names=FALSE, col.names=TRUE)






