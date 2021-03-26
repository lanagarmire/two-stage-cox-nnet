library(tidyverse)  # data manipulation
library(caret)
library(glmnet)
library(survival)
library(dplyr)
library(data.table)


# load image data
image_290 = fread("/home/jingzhe/PAGE-Net/data/image_290.tsv")
image_290 = as.data.frame(image_290)
image_290 = image_290[,-1]

# load clinical data
clin_290 = fread("/home/jingzhe/PAGE-Net/data/clin_290.tsv")
clin_290 = as.data.frame(clin_290)
clin_290 = clin_290[,-1]
colnames(clin_290) = c("time","status")
clin_290$time[clin_290$time==0] = 1
ytime_train = clin_290$time
ystatus_train = clin_290$status

# fit coxph - elastic net
y = Surv(clin_290[,1],clin_290[,2])
set.seed(12345)
cv.tr = cv.glmnet(as.matrix(image_290),y,family='cox',alpha=0.9,nfolds = 10)

#======================================

# negative log likelihood
theta<-predict(cv.tr,as.matrix(image_290),s=cv.tr$lambda.min,type='response')
theta = theta[,1]
exp_theta = exp(theta)

N_train = dim(image_290)[1]
R_matrix_train = matrix(0, nrow = N_train, ncol = N_train)
for (i in 1:N_train){
   for (j in 1:N_train){
      R_matrix_train[i,j] = ytime_train[j] >= ytime_train[i]
   }   
}
PL_train = sum((theta - log(exp_theta %*% R_matrix_train)) * ystatus_train)

# negative log likelihood of mean value
PL_mod = 1:dim(image_290)[2]
x_mean = colMeans(image_290)
for (k in 1:dim(image_290)[2]){
   xk_mean = x_mean[k]
   xk_train = image_290
   xk_train[,k] = xk_mean
   theta = predict(cv.tr,as.matrix(xk_train),s=cv.tr$lambda.min,type='response')
   theta = theta[,1]
   exp_theta = exp(theta)
   PL_mod[k] = sum((theta - log(exp_theta %*% R_matrix_train)) * ystatus_train)
}

# likelihood difference
score = PL_train - PL_mod

# save outputs
table = data.frame(colnames(image_290), score)
colnames(table) = c("feature","score")
write.table(table[order(-table$score),],"coxph_importance.csv",quote=FALSE,sep=',',col.names=TRUE,row.names=FALSE)


