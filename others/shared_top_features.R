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

# load image importance score (seed = 11)
image_importance = fread("/home/jingzhe/cox_nnet/feature_importance_allsample.csv")
image_importance = data.frame(colnames(image_290),image_importance$V1)
colnames(image_importance) = c("feature","score")
image_importance = image_importance[order(-image_importance$score),]

# select coxnnet features
features_coxnnet = image_importance$feature[1:100]
write.csv(image_importance[1:100,], "coxnnet_top100imaging.csv",row.names=FALSE,col.names=TRUE,quote=FALSE)
cat("number of features selected by coxnnet : ",length(features_coxnnet))
cat("\n")

# load clinical data
clin_290 = fread("/home/jingzhe/PAGE-Net/data/clin_290.tsv")
clin_290 = as.data.frame(clin_290)
clin_290 = clin_290[,-1]
colnames(clin_290) = c("time","status")
# replace time=0 with time=1
clin_290$time[clin_290$time==0] = 1

features_all = c()
y = Surv(clin_290[,1],clin_290[,2])
flds = createFolds(1:dim(image_290)[1],5) # 5 folds cross-validation 
count=0
for (fold in flds){
   count = count + 1
   print(count)
   testSet = fold
   trainingSet = which(!(1:dim(image_290)[1] %in% testSet))

   # conventional coxph
   model = cv.glmnet(as.matrix(image_290[trainingSet,]),y[trainingSet],family='cox',alpha=0.9,nfolds = 10)
   cf = coef(model, s = 'lambda.min')
   cc2 = as.matrix(cf[-1,])
   cc2 = data.frame(cc2)
   colnames(cc2)[1] = "V1"
   cc2 = cc2 %>% filter(cc2$V1>0)
   features_cox = rownames(cc2)
   cat("number of features selected by conventional coxph : ",length(features_cox))
   cat("\n")
   features_all = c(features_all,features_cox)
}

print(table(features_all))

final_features=c()
tmp = unique(features_all)
for (feature in tmp){
   if(sum(features_all==feature)>0){
      final_features=c(final_features,feature)
   }
}
print(length(final_features))
# common features
# cat("number of features in common : ", sum(features_all %in% features_coxnnet))
cat("number of features in common : ", sum(final_features %in% features_coxnnet))
cat("\n")



