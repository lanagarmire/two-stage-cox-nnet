#  Two-stage cox-nnet

This is the package of 2-stage coxnnet, which is an extension of original coxnnet developed by Travers. (https://github.com/traversc/cox-nnet)

### run coxnnet on imaging data
* go to folder coxnnet/coxnnet_image/, and run coxnnet_image.py

### run coxnnet on gene expression data
* go to folder coxnnet/coxnnet_rna/, and run coxnnet_rna.py

### run two-stage coxnnet on image and gene expression data
* go to folder coxnnet/two_stage/
* first run coxnnet_hiddenlayer_image.py and coxnnet_hiddenlayer_rna.py respectively
* then run integration.R to integrate the hidden nodes
* finally run coxnnet_image_rna.py to train 2nd stage model

### Package Requirements
* Python 2.7 
* Theano 1.0.0
* Run faster on GPU
