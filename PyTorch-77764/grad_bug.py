import torch
# device = torch.device("mps")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

A = torch.randn(1000, 10, requires_grad = True)

#CPU: correctly returns non-zero gradients for 1 element in each column of A
out = torch.sum(torch.quantile(A, .5, 0, interpolation = "nearest"))
out.backward()
torch.where(A.grad != 0)

A.grad = None

#MPS: usually returns no non-zero elements
#some seeds give non-zero elements but in the wrong places (always in the first row)
out = torch.sum(torch.quantile(A.to(device), .5, 0, interpolation = "nearest"))
out.backward()
torch.where(A.grad != 0)