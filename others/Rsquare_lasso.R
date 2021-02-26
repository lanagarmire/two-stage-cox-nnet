library(tidyverse)  # data manipulation
library(caret)
library(glmnet)
library(survival)
library(data.table)
library(dplyr)

# load image data
image_290 = fread("/nfs/home/jingzhe/PAGE-Net/data/image_290.tsv")
image_290 = as.data.frame(image_290)
image_290 = image_290[,-1]

# load image importance score  (use seed=11)
image_score = fread("/nfs/turbo/umms-garmire/jingzhe/two-stage-cox-nnet/others/data/feature_importance_allsample.csv")
image_score = as.data.frame(image_score)
colnames(image_score) = c("score")
image_score = data.frame(colnames(image_290), image_score[,1])
colnames(image_score) = c("feature","score")
image_score = image_score[order(-image_score$score),]

# select top20 image features
top20_img = image_score$feature[1:20]
# subset image data to top100
image_290_top = image_290[,top20_img]

# load rna data
rna_290 = fread("/nfs/home/jingzhe/PAGE-Net/data/rna_290.tsv")
rna_290 = as.data.frame(rna_290)
rna_290 = rna_290[,-1]

# fit linear regression (y=image featuers ; x=gene features)
rSq = c()
for (i in 1:ncol(image_290_top)){
  y = image_290_top[,i]
  #set.seed(2)
  set.seed(123)
  model = cv.glmnet(as.matrix(rna_290),y,family="gaussian",alpha = 1, nfolds = 5)
  cf = coef(model, s = 'lambda.min')
  #cf = coef(model, s = 'lambda.1se')
  cc2 = as.matrix(cf[-1,])
  predict = as.matrix(rna_290) %*% cc2
  err = predict - y
  r2 = 1-var(err)/var(y)
  rSq = c(rSq,r2)
}
print(summary(rSq))
res = data.frame(colnames(image_290_top), rSq)
colnames(res) = c("feature","rsquare")
print(res[order(-res$rsquare),])





