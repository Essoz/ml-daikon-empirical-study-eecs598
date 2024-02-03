import torch

def invariant_check(input_tensor, q):
    # Calculate the expected result using a reference implementation
    reference_result = torch.quantile(input_tensor, q)
    
    # Run torch.quantile on the MPS device
    mps_result = torch.quantile(input_tensor.to('mps'), q)
    
    # Check if the MPS result matches the reference result within a tolerance
    tolerance = 1e-6  # Adjust the tolerance as needed
    if torch.allclose(mps_result, reference_result, rtol=tolerance, atol=tolerance):
        return True
    else:
        return False

# Example usage:
input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
q = 0.5
is_invariant_valid = invariant_check(input_tensor, q)

if is_invariant_valid:
    print("Invariant check passed.")
else:
    print("Invariant check failed.")
