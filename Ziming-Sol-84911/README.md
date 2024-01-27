# PyTorch-FORUM84911

Issue: Depth of the model
* All frozen -> weak nolinearity -> too shallow -> underfitting
* 0 forzen -> Imagenet 1000 classes vs Cifar 100 classes -> too deep -> overfitting
* Here only freeze 1/4 tunable NN layers (excluding batchnorm layers)

Experiments: Adjusting the number of active layers
* All active: Converges @around 10 epoch, Acc:0.4
* 1/4 active: Converges @around 30 epoch, Acc: 0.62
* 1/6 active: Still not converge @ 40 epoch, Acc>0.63
* 2 layer active: Converges @around 20 epoch, Acc: 0.41
* All forzen: Converges @ first epoch (not trained), Acc: 0.1

## Revelation (from GPT)

Could we classify this as a silent error in pytorch transfer learning pipeline if the number of active/forzen layer is not adjusted properly? Is there a way to autocapture this latent error?

This issue isn't necessarily a "silent error" in the PyTorch transfer learning pipeline. Rather, it's a matter of model tuning and understanding the implications of freezing layers in the context of your specific task. The number of layers to freeze or activate during transfer learning is largely dependent on the similarity between the original task (on which the pre-trained model was trained) and the new task.

However, it's true that improper adjustment can lead to suboptimal results, as shown in the experiments. This isn't an error that can be automatically captured because it's not a programming error or a violation of the PyTorch API. It's a modeling decision that can affect the performance of the model.

To mitigate this issue, you could implement a systematic approach to experiment with different configurations (like a grid search or a random search over the number of layers to freeze), and then choose the configuration that performs best on a validation set. This would allow you to empirically determine the optimal number of layers to freeze for your specific task.

In addition, monitoring the training process (e.g., by logging the loss and accuracy after each epoch) can help identify if the model is underfitting or overfitting, and adjustments can be made accordingly. 


## Useful Resources to look into:

* Image classification via fine-tuning with EfficientNet #Transfer Learning from Pre-trained Weights: https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
* [Google Colab] EfficientNet_Cifar100_finetuning.ipynb https://colab.research.google.com/github/pytorch/ignite/blob/master/examples/notebooks/EfficientNet_Cifar100_finetuning.ipynb#scrollTo=hP_tseP1sXpl
