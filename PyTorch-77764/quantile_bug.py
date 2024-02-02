import torch
# device = torch.device("mps")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device.type)
A = torch.randn(1000, 10)

#need to use "nearest", "lower", or "higher" interpolation methods
#"linear" and "midpoint" fail due to operator `aten::lerp.Tensor_out` not being implemented for MPS
#when dim=None, CPU and MPS give same result
torch.quantile(A, .5, interpolation = "nearest")
torch.quantile(A.to(device), .5, interpolation = "nearest")

#when dim is not None, MPS gives nonsense result
torch.quantile(A, .5, 0, interpolation = "nearest")
torch.quantile(A.to(device), .5, 0, interpolation = "nearest")

#MPS output is same as middle row of tensor
#so torch.quantile must just be pulling 500th index without sorting first
A[500]

#inelegant workaround is to sort the tensor before calling quantile
torch.quantile(torch.sort(A.to(device), dim = 0).values, .5, 0, interpolation = "nearest")