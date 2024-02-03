import torch

def test_matmul_bug():
    zeros = torch.zeros(911, 9, 1, device=torch.device("cpu"))
    ones = torch.ones(1, 32769, device=torch.device("cpu"))
    result_cpu = zeros @ ones

    zeros_mps = torch.zeros(911, 9, 1, device=torch.device("mps"))
    ones_mps = torch.ones(1, 32769, device=torch.device("mps"))
    result_mps = zeros_mps @ ones_mps

    if not torch.allclose(result_cpu, result_mps):
        raise AssertionError("Results are different between CPU and MPS devices")

test_matmul_bug()