# DeepSpeed `BF16_Optimizer` Gradient Clipping Leading to Model Weights Out-of-Sync

The fix for this bug is available in the [PR 1801](https://github.com/microsoft/DeepSpeed/pull/1801), specifically the commit [e24814a10de04ce280efe2adb027b023e3336493](https://github.com/microsoft/DeepSpeed/pull/1801/commits/e24814a10de04ce280efe2adb027b023e3336493).

The bug was originally reported in the Bloom-176B model training process, where the model norm weights were out-of-sync when using the `BF16_Optimizer` with **gradient clipping** and **tensor model parallelism (TP rank > 0)**.

**The cost of this bug is high**, as it leads to incorrect model weights. Though the developers were able to detect the bug in time, fixing its impact (restore the divergent model weights) took a significant amount of time.

## Environment Setup

**Pre-requisites**: You need to have at least two GPUs to reproduce the bug.

**Note**: You are advised to use a virtual environment to install the required dependencies.

1. First install DeepSpeed version 0.6.2 locally in the editable fashion.

   ```bash
   git clone https://github.com/microsoft/DeepSpeed.git
   cd DeepSpeed
   git checkout v0.6.2
   pip3 install -r requirements/requirements.txt
   pip3 install -e .
   ```

   Note: If you still face dependency issues, you can use the `install.sh` script to install and check for the required dependencies.

2. Download bigscience/Megatron-DeepSpeed.

   ```bash
   git clone https://github.com/bigscience-workshop/Megatron-DeepSpeed.git
   git checkout thomas/test_different_layer_norm
   ```

   Install the following dependencies by `pip install -r requirements.txt`. **Do comment out the `deepspeed` line in the `requirements.txt` file before you proceed**.

3. Install CUDA nvcc compiler (this is required for `apex`).
   1. Please check the CUDA version shipped with your PyTorch installation.
   2. Download and the install the CUDA nvcc compiler **with the same CUDA version as PyTorch's**. You can conveniently install it using the script here: [https://github.com/TimDettmers/bitsandbytes/blob/main/install_cuda.sh](https://github.com/TimDettmers/bitsandbytes/blob/main/install_cuda.sh)

4. Install the `apex` library.
   Please go to the [NVIDIA Apex](https://github.com/NVIDIA/apex) repository and follow the installation instructions.

5. Install the `pytest` library.

   ```bash
   pip install pytest
   ```

6. Modify the `deepspeed` library to reproduce the bug.
   1. Open the `deepspeed/runtime/bf16_optimizer.py` file.
   2. Replace the `get_grads_for_norm` function with the following code:

   ```python
    @torch.no_grad()
    def get_grads_for_norm(self, for_clipping=False):
        grads = []
        tensor_mp_rank = bwc_tensor_model_parallel_rank(mpu=self.mpu)
        for i, group in enumerate(self.bf16_groups):
            for j, lp in enumerate(group):
                if not for_clipping:
                    if hasattr(lp, PIPE_REPLICATED) and lp.ds_pipe_replicated:
                        continue

                if not (tensor_mp_rank == 0 or is_model_parallel_parameter(lp)):
                    continue # YUXUAN: as compared to the original code, this line is moved out by one indentation level

                if not self.fp32_groups_has_gradients[i][j]:
                    continue

                grads.append(self.fp32_groups_gradients[i][j])

        return grads
    ```

## Run

We run the developer's test to reproduce the bug.

1. Go to the `Megatron-DeepSpeed` directory.

   ```bash
   cd Megatron-DeepSpeed
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   ```

2. Run the test.

   ```bash
    pytest tests/test_training.py::MegDSTestTraining::test_layer_norm_consistent_0_bf16
    ```

    The test should fail, and you should see an error message similar to the following:

    ```bash
    FAILED tests/test_training.py::MegDSTestTraining::test_layer_norm_consistent_0_bf16 - AssertionError: Checking Transformer Layer norm weights in key input_layernorm.weight, checkpoint global_step10, files ['layer_03-model_00-model_states.pt', 'layer_03-model_01-model_states.pt']
    ```

## What to expect

Model norm weights should be in sync across all the model replicas after sync. When we check the checkpoint files dumped from different model replicas, the model norm weights should be the same. However, the bug causes the model norm weights to be out-of-sync.

**You should expect the test to pass if you undo step 6 of the environment setup.**

## What is the bug

The bug is in the `deepspeed/runtime/bf16_optimizer.py` file. The `get_grads_for_norm` function is not correctly handling the tensor model parallelism (TP rank > 0) when using the `BF16_Optimizer` with **gradient clipping**.

## Root Causes

TODO

## How to fix

The fix was simple, but I haven't quite understood the debugging process yet and the exact root cause of the bug. The fix was to move the `if not (tensor_mp_rank == 0 or is_model_parallel_parameter(lp))` line in by one indentation level.

## Potential Ways to Detect the Bug Automatically

TODO.