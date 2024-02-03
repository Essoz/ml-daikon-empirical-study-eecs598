# PYTORCH-77764 MPS Issues

https://github.com/pytorch/pytorch/issues/101878

https://github.com/pytorch/pytorch/issues/111634

## Environment Setup

## Run
Check in the issue folder
`pwd`
`*/PyTorch-77764/`
Run `pytest` **on a Mac OS Device that supports MPS**

## What is the bug
* incorrect return value for `torch.quantile` on mps device
* incorrect return gradients by the backward method on mps device
* incorrect return for matmul operation on mps device (should be a all-zero matrix but element '1' occurs)

## Possible Cause
* `t.transpose()` and `t.unsqueeze()` functions inside `torch.quantile` provide a strided view that the sort() MPS implementation doesn't handle well
* MPS framework related

## How to Fix
TBD

## Potential Ways to Detect the Bug Automatically
* create unit tests/logging/monitoring that specifically target the behavior of `torch.quantile` when running on MPS devices
* Invariant check:
  * [`test_quantile_inv.py`](/Invariant_check/test_quantile_inv.py):
  Accuracy check: the invariant check compares the MPS result with the reference result using torch.allclose to check if they are close within a specified tolerance.
  * [`test_matmul_inv.py`](/Invariant_check/test_matmul_inv.py):
  Consistency check: the invariant check considers the bug if the number of ones in the result tensor varies across runs.  
