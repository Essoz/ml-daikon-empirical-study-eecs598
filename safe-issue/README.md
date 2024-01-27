# PyTorch-FORUM84911

Issue: Depth of the model
* All frozen -> weak nolinearity -> too shallow -> underfitting
* 0 forzen -> Imagenet 1000 classes vs Cifar 100 classes -> too deep -> overfitting
* Here only freeze 1/4 tunable NN layers (excluding batchnorm layers)

Useful Resources to look into:
* Image classification via fine-tuning with EfficientNet #Transfer Learning from Pre-trained Weights: https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
* [Google Colab] EfficientNet_Cifar100_finetuning.ipynb https://colab.research.google.com/github/pytorch/ignite/blob/master/examples/notebooks/EfficientNet_Cifar100_finetuning.ipynb#scrollTo=hP_tseP1sXpl
