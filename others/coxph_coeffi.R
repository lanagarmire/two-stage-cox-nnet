library(survival)
library(data.table)

# load clinical info
clin_290 = fread("clin_290.tsv")
clin_290 = data.frame(clin_290)
clin_290 = clin_290[,-1]
colnames(clin_290) = c("time","status")
# clin_290$status = ifelse(clin_290$status==1,2,1)

# load image features
image_290 = fread("image_290.tsv")
image_290 = data.frame(image_290)
image_290 = image_290[,-1]

# load gene features
rna_290 = fread("rna_290.tsv")
rna_290 = data.frame(rna_290)
rna_290 = rna_290[,-1]

dat = data.frame(clin_290, image_290, rna_290)
covariates = colnames(dat)[3:ncol(dat)]

# fit single-variate coxph model
univ_formulas <- sapply(covariates,
                        function(x) as.formula(paste('Surv(time, status)~', x)))
univ_models <- lapply( univ_formulas, function(x){coxph(x, data = dat)})
univ_results <- lapply(univ_models,
                       function(x){ 
                          x <- summary(x)
                          p.value<-signif(x$wald["pvalue"], digits=2)
                          wald.test<-signif(x$wald["test"], digits=2)
                          beta<-signif(x$coef[1], digits=2);#coeficient beta
                          res<-c(beta, p.value)
                          names(res)<-c("beta", "pvalue")
                          return(res)
                          })
res = data.frame(names(univ_results),unlist(univ_results, use.names=FALSE))
#res = t(as.data.frame(univ_results, check.names = FALSE))
#res = as.data.frame(res)
colnames(res) = c("covariates","sign")
res$sign = ifelse(res$sign>0,"pos","neg")
write.csv(res, "coxph_coeffi.csv", row.names=FALSE, col.names=TRUE, quote=FALSE)






