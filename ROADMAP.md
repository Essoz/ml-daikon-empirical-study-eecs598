# Roadmap of *Machine Learning Issues*

From a high-level perspective, this project aims to help developers detect silent/latent correctness issues (bugs) in machine learning pipelines before those issues result in unbearable costs.

This file documents the storyline of the project, the milestones, and the goals of each milestone. It is a living document that will be updated as the project progresses.

## 1. Key Research Questions

The following are the key research questions that we will answer in this project:

1. What are the silent/latent correctness issues of machine learning pipelines?

    Reasons why this question is important:
    1. The insights from this question will help us understand the problem space and guide our design decisions.
    2. The insights might also help future researchers to design better abstractions to prevent those issues from happening in the first place.
2. How can we detect those silent/latent correctness issues early on?

    Reasons why this question is important:
    1. A tool that can detect those issues early on will help developers save a lot of time and money.

The meaning of "silent/latent" is that the issues are either completely silent (i.e., no error message is reported) or only raise exceptions when the pipeline is executed for a long time (e.g., hours).

For now, we are focusing on user level issues (i.e., issues that are caused by the user code or the user data). We are not focusing on issues that are caused by the environment (e.g., GPU drivers, CUDA, etc.) or issues that are caused by the ML library itself (e.g., PyTorch, TensorFlow, etc.).

Please see [CONTRIBUTING.md#bug-selection](https://github.com/OrderLab/machine-learning-issues/blob/main/CONTRIBUTING.md#bug-selection) for more details.

### What are not (necessarily) the scope of this project?

1. **Hyperparameter tuning**: This is a very important yet heavily studied problem. As a result, we want to focus on other problems that are less studied but still matters a lot.
2. **Solving environment issues**: Engineering-wise, this is a very hard problem to solve due to the complexity of the Python ecosystem, lack of standardization in the ML community, and the lack of control over the user's environment. We feel like this problem is better solved by the ML community as a whole or by the Software Engineering community.
3. **Issues that are not silent/latent**: Issues that raise exceptions early on are usually easy to detect and fix. Though they still troubles developers a lot, the cost of such issues and effort to debug them are usually not as high as silent/latent issues.
4. **Finding issues in GPU Drivers, CUDA, or other non-user-level code**: Quality assurance of those components is very important, and our project should be able to help with that. **Our technique is indeed aimed at detecting general silent/latent correctness issues in ML pipelines due to various root causes despite their location no matter it is in user level code or the environment. However, we choose to focus on user issues for now as they are easier to control.** For example, a PyTorch API raising an exception might be caused by a bug in PyTorch, a bug in CUDA, or a bug in the user code or an environment installation issue. As a result, we feel like for now it is better to focus on user issues as they are easier to control. We might consider expanding our scope in the future.

## 2. Introduction & Expected Contribution

<!-- TODO: cite blogs, papers (general case points) -->
As machine learning (ML) gains popularity in both academia and industry, the correctness of ML pipelines becomes increasingly important. A silent or latent correctness issue in an ML pipeline can result in a huge cost, especially when the pipeline is executed for a long time (e.g., hours). The cost is further amplified as the scale of ML pipelines increases astronomically with the recent LLM arms race [1]. Detecting such issues and fixing them introduces a huge burden on developers. For example, a developer might need to wait for hours to see if the pipeline raises an exception or not. Even worse, the pipeline might not raise an exception at all, but the results are completely wrong. Also even if the bug has been detected, it is also hard to estimate the time needed to fix it. For example, a developer might need to spend hours to find out that the bug is caused by a missing data preprocessing step. As a result, a tool that can detect such issues early on will help developers save a lot of time and money. TODO: cite tensor shape mismatch paper, BLOOM training blog, etc.

<!-- What's the challenges of finding silent/latent correctness issues? -->
Silent or latent correctness issues are easy to make and hard to detect. A few hypothesis that they are easy to make are:

1. Flexible and dynamic nature of ML pipelines. Thus, it is harder to formally control the quality of ML pipelines with tools like git. TODO: CMU SEI Blog
2. Lack of formal specifications of ML libraries. Thus, it is harder to formally verify the correctness of ML pipelines.
3. Long execution time. Thus, it is harder to detect issues early on.
4. Data-dependent nature of ML pipelines. Thus, it is harder to apply traditional software engineering techniques (e.g., unit testing) to ML pipelines.

Additionally, unlike traditional software, correctness issues might not only be caused by the code but also by the data. For example, a data preprocessing step might be inappropriate (e.g. [PyTorch#FORUM84911](https://github.com/OrderLab/machine-learning-issues/blob/main/PyTorch-FORUM84911/README.md#pytorch-forum84911)), or the data might be corrupted.

<!-- What's missing in existing work, from a high-level perspective? -->
While there are existing work that can help developers detect issues in ML pipelines, they fall short in the following ways:
1. Existing work focuses on ensuring quality of ML models (e.g., hyperparameter tuning, model debugging, deepXplore[4], etc.) and DL frameworks (e.g., [3]) rather than the quality of ML pipelines
2. Existing work focusing on ML pipelines focuses on a very specific type of issues (e.g., tensor shape mismatch[5]) or are limited cause they need prior runs of the same pipeline to pinpoint the issue (e.g., ).
3. Existing work struggle to scale beyond simple ML pipelines (due to the enormous amount of Python syntax and API to support and the inherent limitation of static analysis).


<!-- What's our approach? i.e. our hypothesis -->
We want to create a general-purpose tool that can detect a wide range of silent/latent correctness issues in ML pipelines. Our high-level approach is to leverage dynamic analysis to infer the likely invariants of ML pipelines and then use those invariants to detect correctness issues.

A few directions we are considering:

1. **Repository Mining + Static Analysis**:
Infer library API specifications (pre-conditions and post conditions) from examples (e.g [PyTorch/examples](https://github.com/pytorch/examples)\[2\]). Then, use the inferred specifications to statically verify the user code connecting two API calls in the new ML pipeline.
    - Pro: The inferred specifications are likely to be correct and comprehensive as they are mined from a huge number well-maintained examples.
    - Con: The inferred invariants might be best for detecting issues that raises exceptions. It might not be as good for detecting issues that are completely silent as these are usually caused by the data or inappropriately configuration parameters.

2. **Predictive Analysis**: The hypothesis is that although ML pipelines are expensive to run, we can use a small number of executions to predict the correctness of future executions.
    - Pro: The cost of running ML pipelines is reduced.
    - Con: The correctness of the predictions is not guaranteed.

3. **Online Anomaly Detection**: The hypothesis is that the silent/latent correctness issues will cause the execution of the ML pipeline to violate certain invariants. We can use online anomaly detection to detect such violations.

<!-- What's our expected contribution? -->
We expect to make the following contributions:

1. In-depth understanding of the silent/latent correctness issues of ML pipelines.

2. A tool that can detect a wide range of correctness issues in ML pipelines at compile time or early on during runtime.

## 3. Stages

1. Understanding Real-World Silent/Latent Correctness Issues [*Current Stage*]
    - [x] Collecting real-world issues
        - [x] Collecting issues from GitHub issues
        - [x] Collecting issues from StackOverflow
        - [x] Collecting issues from other sources (e.g., Reddit, Twitter, etc.)
    - [x] Analyzing the issues
    - [x] Summarizing the issues
2. Testing the Hypothesis [*Current Stage*]
    - [ ] Testing all the hypothesis we have
        - [ ] Implementing prototypes
        - [ ] Evaluating prototypes
3. Iterating on the Hypothesis
    - [ ] Iterating on the hypothesis
        - [ ] Desiding on one potential direction to focus on
        - [ ] Finish iterating on the direction with data collected from Stage 1
        - [ ] Repeating Stage 2 and 3
    - [ ] Large-Scale Evaluation
        - [ ] Collecting more real-world issues (probably from industry collaborators)
        - [ ] Evaluating the tool on the issues
4. Large-Scale Evaluation
    - [ ] Collecting more real-world issues (probably from industry collaborators)
    - [ ] Evaluating the tool on the issues
5. Writing Paper
    - [ ] Writing the paper
    - [ ] Submitting the paper

## 4. Milestones

### 4.1. Milestone 1: Understanding Real-World Silent/Latent Correctness Issues

- **Outcome**: A list of 10 real-world silent/latent correctness issues that 1) meet our bug-choosing criteria, 2) are reproducible, and 3) are analyzed. We will also analyze the issues and summarize the findings, to guide our design decisions.
- **Satisfying Criteria**: This milestone will be considered as satisfied if we have several issues that meet our bug-choosing criteria, are reproducible, and are analyzed.
- **Deadline**: No deadline. We will keep collecting issues even after this milestone is satisfied.

### 4.2. Milestone 2: Have a working prototype

- **Outcome**: A working prototype built from our hypothesis that can detect a wide range of silent/latent correctness issues in ML pipelines. The minimum requirement is being able to infer certain types of constraints from the pipeline.
- **Satisfying Criteria**:
  - This milestone will be considered as satisfied if the working prototype shows promising results on the issues collected in Milestone 1.
  - A working prototype is considered as "working" if it can detect some silent/latent correctness issues in ML pipelines at a satisfying practicality (e.g., the cost of running the pipeline is not too high).
- **Deadline**: 2024 Winter

### 4.3. Milestone 3: Reaching a satisfying utility

- **Outcome**: A tool that can detect a wide range of silent/latent correctness issues in ML pipelines with a satisfying utility.
- **Satisfying Criteria**: This milestone will be considered as satisfied if the tool can detect a wide range of silent/latent correctness issues in ML pipelines with a satisfying utility. We also need a very large set of real-world issues to evaluate the tool.
- **Deadline**: 2024 Summer

### 4.4. Milestone 4: Writing Paper

- **Outcome**: A paper that describes the tool and the findings.
- **Satisfying Criteria**: This milestone will be considered as satisfied if the paper is accepted by a top-tier conference.
- **Deadline**: 2024 Fall (Before the OSDI deadline)

## 5. Goals by the end of this semester (2024 Winter)

- Finish Milestone 1 and 2.
- Have a plan for Milestone 3.
- Have a revised introduction, background, and related work section for the paper (Milestone 4).

## 6. Existing Flaws
<!-- Mostly on understanding the issues and how far has the research community gone. -->

Since this project aims to leverage dynamic invariant inference to detect silent/latent correctness issues in ML pipelines, the following are the existing flaws that we need to address threats to validity related to this approach:

1. Input:
    - Whether the input (e.g. collected trace, ml pipelines) is representative of real-world ML pipelines?
    - Whether the input itself is correct? If not, the invariants inferred from the input might be incorrect.
    - Noise in the input (e.g. irrelevant variables, or indeterminism, etc.) might affect the quality of the inferred invariants.
2. Performance:
    - Whether the performance of the tool is acceptable? If not, the tool might not be practical.
3. Accuracy:
    - Whether the inferred invariants are accurate? If not, the tool might not be able to detect the issues.

## References

1. [Unicron: Economizing Self-Healing LLM Training at Scale](https://arxiv.org/abs/2401.00134)
2. [PyTorch/examples](https://www.github.com/pytorch/examples)
3. [Fuzzing Deep-Learning Libraries via Automated Relational API Inference](https://cs.stanford.edu/~anjiang/papers/FSE22DeepREL.pdf)
4. [DeepXplore: Automated Whitebox Testing of Deep Learning Systems](https://arxiv.org/abs/1705.06640)
5. [A static analyzer for detecting tensor shape errors in deep neural network training code](https://arxiv.org/abs/2112.09037)
6. [Debugging Machine Learning Pipelines](https://arxiv.org/abs/2112.09037)
