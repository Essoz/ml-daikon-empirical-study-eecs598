# Guidelines for adding bugs to the list

## Bug Selection

You can find bugs from the following source:

- GitHub Issues/PRs/Commits
- DL Framework Forums(e.g. [PyTorch's Forum](https://discuss.pytorch.org/))
- QA Forums(e.g. [StackOverflow](https://stackoverflow.com/))
- Research Papers (see [OrderLab/awesome-machine-learning-reliability](https://github.com/OrderLab/awesome-machine-learning-reliability))

### Standard for choosing bugs

We want to find bugs that are machine learning related. Such bugs should have the following characteristics:

- **Silent**: The bug does not cause any error or warning in initial runs. It may stay completely silent and only produce incorrect results.
- **Hard to detect**: The bug might be flaky, only occur in certain environments, or only occur when encountering some special inputs (e.g. NaN bugs).
- **Hard to debug**: The exception message might not point directly to the root cause of the bug, and it needs some extra effort to debug.
- **Costly**: The bug is costly. For example, a bug in validation code can cause all prior training to be wasted.

The above list is not exhaustive and each point can have some overlap. As long as you find a bug interesting, feel free to add it to the list. When unsure, **please open an issue to discuss**.

#### Some examples of interesting bugs 

TBD @essoz

#### What are the kinds of bug that we are not interested in?:

We are not interested in bugs that are only related to the implementation of the framework or related to Python itself. For example, a buggy framework implementation is not interesting. A pipeline that fails to run due to incorrect library installation is not interesting.

## Bug Folder Structure

Please name your bug folder as `<framework>-<bug-id>`, e.g. `pytorch-1`. The id should be the GitHub issue number or PR number. If the bug is not from GitHub, please use the following format: `<framework>-<forum>-<bug-id>`, e.g. `pytorch-stackoverflow-1`.

The folder should contain the following files:

- `README.md`: A write-up of the bug.
- `bug.py`: The code to reproduce the bug.
- `bug-fix.py`: The code that has the bug fixed.
- `environment.yml`: The conda environment file to reproduce the bug's environment (if some dependencies cannot be installed via conda, please describe in README).

## Bug Write-up (README.md)

To add a bug to the list, please follow the following format:
```markdown
# <Bug Title>

<URL to the bug>

## Environment Setup

Describe how to setup the environment to reproduce the bug.

## Run

Describe how to run the code to reproduce the bug.

## What to expect

Describe what to expect during and after execution.

## What is the bug

Describe the bug in detail.

## Root Causes

Describe the root causes of the bug.

## How to fix

Describe how to fix the bug.

## Potential Ways to Detect the Bug Automatically

Describe how can we detect the bug automatically.

```
