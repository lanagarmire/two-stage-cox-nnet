library(tidyverse)  # data manipulation
#library(caret)
library(glmnet)
library(survival)
library(data.table)
library(dplyr)

# extract overall pvalue from lm fit
lmp <- function (modelobject) {
    if (class(modelobject) != "lm") stop("Not an object of class 'lm' ")
    f <- summary(modelobject)$fstatistic
    print(f)
    p <- pf(f[1],f[2],f[3],lower.tail=F)
    attributes(p) <- NULL
    return(p)
}

# load image data
image_290 = fread("/nfs/home/jingzhe/PAGE-Net/data/image_290.tsv")
image_290 = as.data.frame(image_290)
image_290 = image_290[,-1]

# load image importance score  (use seed=11), computed by training all samples
image_score = fread("/nfs/turbo/umms-garmire/jingzhe/two-stage-cox-nnet/others/data/feature_importance_allsample.csv")
image_score = as.data.frame(image_score)
colnames(image_score) = c("score")
image_score = data.frame(colnames(image_290), image_score[,1])
colnames(image_score) = c("feature","score")
image_score = image_score[order(-image_score$score),]

# select top20 image features
#top20_img = image_score$feature[1:20]
top20_img = c('Granularity_11_MaskedEWithoutOverlap',
              'Mean_FilteredNuclei_AreaShape_Orientation',
              'Granularity_5_MaskedEWithoutOverlap',
              'ImageQuality_MaxIntensity_MaskedHWithoutOverlap',
              'Intensity_MaxIntensity_MaskedHWithoutOverlap',
              'StDev_Cells_Neighbors_PercentTouching_Expanded',
              'Mean_Tissue_Location_Center_Y',
              'Mean_Nuclei_AreaShape_Orientation',
              'Mean_Nuclei_Texture_SumEntropy_MaskedHWithoutOverlap_3_0',
              'Median_Nuclei_Texture_SumEntropy_MaskedHWithoutOverlap_3_135') 
# subset image data to top100
image_290_top = image_290[,top20_img]

# load rna data
rna_290 = fread("/nfs/home/jingzhe/PAGE-Net/data/rna_290.tsv")
rna_290 = as.data.frame(rna_290)
rna_290 = rna_290[,-1]

# fit linear regression (y=image featuers ; x=gene features)
image_features = c()
rSq = c()
pvals = c()
for (i in 1:ncol(image_290_top)){
  #print(i)
  y = image_290_top[,i]
  #set.seed(2)
  set.seed(123)
  model = cv.glmnet(as.matrix(rna_290),y,family="gaussian",alpha = 1, nfolds = 5)
  cf = coef(model, s = 'lambda.min')
  #cf = coef(model, s = 'lambda.1se')
  cc2 = as.matrix(cf[-1,])
  #predict = as.matrix(rna_290) %*% cc2
  #err = predict - y
  #r2 = 1-var(err)/var(y)
  #rSq = c(rSq,r2)
  selected_genes = which(cc2!=0)
  if(length(selected_genes)==0) next
  image_features = c(image_features,colnames(image_290_top)[i])
  if(length(selected_genes)==1){dat = cbind(y, rna_290[selected_genes])}
  else{dat = cbind(y, rna_290[,selected_genes])}
  names(dat)[1] = 'image'
  fit = lm(image~.,dat)
  rSq = c(rSq, summary(fit)$r.squared)
  pvals = c(pvals, lmp(fit)) 
}
#print(summary(rSq))
res = data.frame(image_features, rSq, pvals)
colnames(res) = c("feature","rsquare","pvalue")
res = res[order(-res$rsquare),]
print(res)
write.csv(res,"rsquare_table.csv",row.names=FALSE,quote=FALSE)




