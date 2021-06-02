#  Two-stage cox-nnet

This is the package of 2-stage coxnnet, which is an extension of original coxnnet developed by Travers. (https://github.com/traversc/cox-nnet)

### run coxnnet on imaging data
* go to coxnnet/coxnnet_image/, and run coxnnet_image.py

### run coxnnet on gene expression data
* go to coxnnet/coxnnet_rna/, and run coxnnet_rna.py

### run two-stage coxnnet on image and gene expression data
* go to coxnnet/two_stage/, then 

* Train Cox-nnet on image data (*1st phase*)
* Train Cox-nnet on gene data (*1st phase*)
* Hidden nodes integration of two data types
* Train Cox-nnet on integrated hidden features (*2nd phase*)

### Package Requirements
* Python 2.7 
* Theano 1.0.0
* Run faster on GPU
