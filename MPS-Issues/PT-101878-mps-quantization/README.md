# torch.quantile on MPS doesn't sort values when dim is not None

https://github.com/pytorch/pytorch/issues/101878

## Environment Setup

This bug has been verified to reproduce on 
- Macbook Air M2, Sonoma
- Macbook Pro M3 Pro, Sonoma
and on the latest PyTorch version (2.2.2)

```shell
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2
```


## Run

The file `test_quantile_bug.py` has three tests to indicate under what scenarios does the bug occur. To run the test, simply run 

```
python3 test_quantile_bug.py
```

## What to expect

The script executes three tests which exercises `torch.quantile` with different inputs. Specifically there are three tests:

1. `test_quantile_dim_none` that executes `torch.quantile` with `dim=None` both on CPU and MPS. 
2. `test_quantile_dim_zero` that executes `torch.quantile` with `dim=0` both on CPU and MPS.
3. `test_quantile_sorted_tensor` that executes `torch.quantile` with `dim=0` but the input already sorted.

In test 1 and 3, the bug is not triggered and thus will pass. Test 2 will fail.

## What is the bug

[`torch.quantile`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Sorting.cpp#L689) does not give correct results on MPS (Metal Performance Shader) when the input tensor is not sorted already and `dim` is not None.

This bug is silent.

## Root Causes

`torch.quantile` relies on [`sort`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/operations/Sort.mm) to find the specific quantile value of a tensor. Users can supply an optional `dim` argument to tell torch along which dimension to calcuate quantile for. 

When `dim` is provided, `torch.quantile` prepares a new sorted tensor using  

```cpp
sorted = std::get<0>(self.unsqueeze(-1).transpose(wrapped_dim, -1).sort());
```

The tensor produced by `self.unsqueeze(-1).transpose(wrapped_dim, -1)` is a "strided" one and may not be correctly handled by `sort` implementation on mps ([mps kernels do not support heterogeneous memory access so pytorch's strided tensor needs special handling on mps](https://github.com/pytorch/pytorch/wiki/MPS-Backend#view-ops-in-mps-using-gather-scatter-approach)).

To reproduce the root cause, simply run the following code and observe the differences:

```python
A = torch.randn(1000, 10).to('mps')
A_view = A.unsqueeze(-1).transpose(0, -1)
print("Strided:", torch.allclose(A_view.sort().values, A_view))
print("Contiguous:", torch.allclose(A_view.clone(memory_format=torch.contiguous_format).sort().values, A_view))
```

## Impact Analysis

This bug lies in the incorrect implementation of the mps `sort`. Thus theoretically, this bug might affect any computation that uses `sort`. 

In practice, sort is not a common op in machine learning, especially training. Sometimes, `sort` might get used in:

1. Data Preprocessing
	1. Data Normalization and Scaling:
	    - Quantile-based Scaling: Quantile Transformer from sklearn.preprocessing scales features to follow a uniform or normal distribution. This method involves sorting the data and then using the empirical cumulative distribution function.
	    - Winsorizing: Limits extreme values in the data by replacing them with a specified percentile value, which requires sorting the data to determine these percentiles.
	2. Outlier Detection and Removal:
        - Quantile Thresholding: Outliers can be detected by computing the quantiles of the data distribution and removing data points that lie below the lower quantile or above the upper quantile.
        - Rank-based Methods: Sorting is used to rank data points, and outliers can be identified based on their ranks.
	3. Data Binning:
        - Quantile Binning: Data is divided into bins such that each bin has the same number of data points. This involves sorting the data and then partitioning it into bins based on quantile thresholds.
2. Ranking-based Evaluation Metrics
	1. Quantile Regression Metrics:
	    - Quantile regression is used to estimate the conditional quantiles of the response variable distribution, making it robust to outliers. The evaluation involves calculating quantile losses, which requires sorting operations.
	2. Ranking Metrics:
	    - NDCG (Normalized Discounted Cumulative Gain): Measures the ranking quality of predictions. Sorting is required to compare the predicted ranking with the ideal ranking.
        - MAP (Mean Average Precision): Often used in information retrieval tasks, it involves calculating the precision at each relevant item in the sorted list of predictions.
	3. Percentile-based Metrics:
        - Metrics such as the 95th percentile of latency or error rates in performance evaluation involve sorting the data to determine the specified percentile value.
3. Really Special Model Architectures

## How to fix

There are a few ways to fix this issue. One way would definitely be fixing the implementation of `sort` directly. The challenge lies in understanding the specific location within `sort` that triggered the bug, as the bug doesn't seem to be triggered by every strided tensor.

An alternative workaround is to make sure the input to any ops that use `sort` is contiguous by using `.clone(memory_format=torch.contiguous_format)`.

## Potential Ways to Detect the Bug Automatically

This bug is a low-level API contract violation. For example, 
- `sort` should return the sorted tensor, but it didn't.
- `torch.quantile` should return the tensor's quantile value, but it didn't.

The end-to-end implication of this bug is unclear as it does not happen in training.

To detect the issue at the API level, we need to be able to infer the input-output constraints for a specific API, and regularly check whether the output of an API call conforms to its input and the constraints.
 