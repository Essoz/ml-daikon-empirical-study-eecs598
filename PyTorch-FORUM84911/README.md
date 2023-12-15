# PyTorch-FORUM84911

https://discuss.pytorch.org/t/obtaining-abnormal-changes-in-loss-and-accuracy/84911


## Setup

1. Create a conda environment
2. Install PyTorch and efficientnet_pytorch
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install efficientnet_pytorch
```

## Run

```bash
python bug.py
```

## What to expect

`bug.py` will train the pretrained model [efficientnet-b0](https://github.com/lukemelas/EfficientNet-PyTorch) on [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html), a dataset of 100 classes of 32x32 images. The training will run for 40 epochs.

After training finishes, the model will dump the loss and accuracy for each epoch into `case_3_res.json`. The model itself will be saved as `case_3_model.pth`.

## What is the bug

- **Model Performance**: As shown from the accuracy and loss log, the model is not learning. The accuracy stays below 1% and the loss does not show a decreasing pattern.
- **Training Speed and Memory Usage**: The training speed is very slow. It takes about 1.5 hours per epoch on a Tesla A40 or a RTX A2000 GPU with batch size equal to 4. The memory usage is also very high. It takes about 11GB GPU memory to train the model with batch size equal to 4. 


## Root Causes

- **All major layers of the model are frozen**. The only trainable layers are the batch normalization layers. The model is not learning because the frozen layers are not updated.
  ```python
  for name,param in model_transfer.module.named_parameters():
      if("bn" not in name):
          param.requires_grad = False

  for param in model_transfer.module._fc.parameters():
      param.requires_grad = False
    
  print(model_transfer.module._fc.in_features)
  ```
  After fixing this issue, the model starts to learn (from ~1% to ~10%).

- **Mismatch between input image resolution and preprocessing rescales**. The input preprocess pipeline rescales the input image to 1024x1024, which does not match the model's input resolution of 224x224. Also, since the CIFAR100 dataset have only 32x32 images, rescaling to 1024x1024 is inappropriate.
  ```python
  train_transforms = transforms.Compose([
      transforms.Resize((1024,1024)),
      transforms.RandomHorizontalFlip(),
      transforms.RandomRotation(10),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], 
                          [0.229, 0.224, 0.225])
  ])
  ```
  The effect of this issue is two fold:
    1. The model's input is not what it expects. The model expects a 224x224 image, but the input is a 1024x1024 image. This will cause the model to perform poorly.
    2. Excessive computation and memory usage. The model's input is 1024x1024, which is roughly 20 times larger than the expected 224x224. This leads to additional computation and memory usage. After fixing this issue, the 1.5 hours per epoch is reduced to 3 minutes per epoch and the memory usage is also significantly reduced.

    **This is the most important issue that needs to be fixed.** After this, the model starts to learn much better (from ~10% to ~40%).

- **Learning Rate and Batch Size**
    - **Learning Rate**: The learning rate is a little large. The learning rate is set to 0.01. 
    - **Batch Size**: The batch size is a little small. The batch size is set to 4.
    Decreasign the learning rate to 0.005 and increasing the batch size to 64 will help the model to learn better (from ~30% to ~45%) and faster.

## How to fix
1. Fix the root causes mentioned above.
    - Set requires_grad to True for all layers.
    - Resize the input image to 224x224.
    - Decrease the learning rate to 0.005.
    - Increase the batch size to 64.