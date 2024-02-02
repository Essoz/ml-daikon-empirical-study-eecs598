# PYTORCH-77764

## quantile_bug

### CPU

``` 
torch.quantile(A, .5, 0, interpolation = "nearest")
tensor([ 0.0279,  0.0380,  0.0013,  0.0168, -0.0601, -0.0600,  0.0022, -0.0169,
        -0.0453, -0.0176])
```

### CUDA

```
torch.quantile(A.to(device), .5, 0, interpolation = "nearest")
tensor([ 0.0279,  0.0380,  0.0013,  0.0168, -0.0601, -0.0600,  0.0022, -0.0169,
        -0.0453, -0.0176], device='cuda:0')
```

## grad_bug

### CPU

```
torch.where(A.grad != 0)
(tensor([ 11,  70, 144, 208, 262, 698, 784, 907, 949, 975]), tensor([9, 8, 3, 7, 4, 5, 2, 0, 1, 6]))
```

### CUDA

```
torch.where(A.grad != 0)
(tensor([ 11,  70, 144, 208, 262, 698, 784, 907, 949, 975]), tensor([9, 8, 3, 7, 4, 5, 2, 0, 1, 6]))
```