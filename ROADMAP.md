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

TODO: add that we focus on Python-based ML pipelines and want to tackle user-level code.

The meaning of "silent/latent" is that the issues are either completely silent (i.e., no error message is reported) or only raise exceptions when the pipeline is executed for a long time (e.g., hours).

Please see [CONTRIBUTING.md#bug-selection](https://github.com/OrderLab/machine-learning-issues/blob/main/CONTRIBUTING.md#bug-selection) for more details.

### What are not (necessarily) in the scope of this project?

1. **Hyperparameter tuning**: This is a very important yet heavily studied problem. As a result, we want to focus on other problems that are less studied but still matters a lot.
2. **Solving environment issues**: Engineering-wise, this is a very hard problem to solve due to the complexity of the Python ecosystem, lack of standardization in the ML community, and the lack of control over the user's environment. We feel like this problem is better solved by the ML community as a whole or by the Software Engineering community.
3. **Issues that are not silent/latent**: Issues that raise exceptions early on are usually easy to detect and fix. Though they still troubles developers a lot, the cost of such issues and effort to debug them are usually not as high as silent/latent issues.
4. **Finding issues in GPU Drivers, CUDA, or other non-user-level code**: Quality assurance of those components is very important. However, we feel like the root cause of such issues is usually diverse and hard to pinpoint. For example, a PyTorch API raising an exception might be caused by a bug in PyTorch, a bug in CUDA, or a bug in the user code or an environment installation issue. As a result, we feel like there is not much we can do to help developers debug such issues.


## 2. Introduction & Expected Contribution

As machine learning (ML) gains popularity in both academia and industry, the correctness of ML pipelines becomes increasingly important. A silent or latent correctness issue in an ML pipeline can result in a huge cost, especially when the pipeline is executed for a long time (e.g., hours). The cost is further amplified as the scale of ML pipelines increases astronomically with the recent progress in LLM research [1].

<!-- What's the challenges of finding silent/latent correctness issues? -->
Silent or latent correctness issues are hard to detect.

1. Flexible and dynamic nature of ML pipelines.
2. Lack of formal specifications.
3. Long execution time.

Additionally, unlike traditional software, correctness issues might not only be caused by the code but also by the data. For example, a data preprocessing step might be missing, or the data might be corrupted.

<!-- How does existing work fall short? -->

1. Existing work focuses on detecting issues in ML models, not ML pipelines.
2. Existing work focusing on ML pipelines focuses on a very specific type of issues (e.g., tensor shape mismatch).
3. Existing work struggle to scale beyond simple ML pipelines (due to the flexibility of Python community and the inherent limitation of static analysis).

TODO: support these points with references or case studies, providing concrete examples of where and how these shortcomings manifest.

<!-- What's missing in existing work, from a high-level perspective? -->

<!-- What's our approach? i.e. our hypothesis -->
We want to create a general-purpose tool that can detect a wide range of silent/latent correctness issues in ML pipelines. Our high-level approach is to leverage dynamic analysis to infer the likely invariants of ML pipelines and then use those invariants to detect correctness issues.

A few directions we are considering:

1. **Repository Mining + Static Analysis**:
Infer library API specifications (pre-conditions and post conditions) from examples (e.g [PyTorch/examples](https://github.com/pytorch/examples)). Then, use the inferred specifications to statically verify the user code connecting two API calls in the new ML pipeline.
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

## 4. Milestones

## 5. Goals by the end of this semester (2024 Winter)

## 6. Existing Flaws

## References

1. [Unicron: Economizing Self-Healing LLM Training at Scale](https://arxiv.org/abs/2401.00134)
2. [PyTorch/examples](https://www.github.com/pytorch/examples)