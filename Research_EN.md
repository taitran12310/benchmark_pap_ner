# Bài gốc: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0353166

# Benchmarking Recent Pre-trained Language Models for Vietnamese Administrative Named Entity Recognition

## 1. Research Background

Named Entity Recognition (NER) is an important Natural Language Processing (NLP) task that identifies and classifies named entities in text.

Vietnamese administrative documents contain domain-specific entities such as government agencies, legal documents, organizations, dates, and quantities. These characteristics make Vietnamese Administrative NER different from general-domain NER.

Recently, La et al. introduced PAP_NER, a large-scale Vietnamese administrative NER corpus, and proposed a hybrid PhoBERT-CRF architecture. The proposed approach achieved a Micro-F1 score of 97.95% on the PAP_NER benchmark.

However, pretrained language models have continued to evolve rapidly. Therefore, it is valuable to investigate whether more recent pretrained language models can improve the performance of the established PhoBERT-CRF baseline on Vietnamese administrative text.

---

# 2. Research Problem

The original PAP_NER study established PhoBERT-CRF as a strong baseline for Vietnamese Administrative NER.

However, the performance of more recent pretrained language models on the PAP_NER dataset has not been sufficiently investigated.

Therefore, this study focuses on benchmarking recent pretrained language models under the same experimental setting as the original PAP_NER study.

The main research problem is:

> Can recent pretrained language models improve Vietnamese Administrative NER performance over the PhoBERT-CRF baseline on the PAP_NER dataset?

---

# 3. Research Objectives

## 3.1. Main Objective

Evaluate the effectiveness of recent pretrained language models for Vietnamese Administrative Named Entity Recognition using the PAP_NER benchmark.

## 3.2. Specific Objectives

1. Reproduce the PhoBERT-CRF baseline reported in the original PAP_NER paper.
2. Select several recent pretrained language models suitable for Vietnamese NLP.
3. Fine-tune the selected models on the PAP_NER dataset.
4. Integrate the selected pretrained language models with the CRF layer.
5. Compare their performance against the original PhoBERT-CRF baseline.
6. Analyze performance for each administrative entity type.
7. Perform error analysis to identify the strengths and weaknesses of each model.
8. Compare model performance and computational cost.

---

# 4. Research Questions

## RQ1

> Can recent pretrained language models outperform the PhoBERT-CRF baseline on the PAP_NER benchmark?

## RQ2

> How does the choice of pretrained language model affect Vietnamese Administrative NER performance across different entity types?

## RQ3

> What types of errors are reduced or introduced when replacing PhoBERT with a more recent pretrained language model?

---

# 5. Research Hypotheses

## H0

> Recent pretrained language models do not significantly improve NER performance compared with the PhoBERT-CRF baseline on PAP_NER.

## H1

> At least one recent pretrained language model significantly improves NER performance compared with the PhoBERT-CRF baseline on PAP_NER.

---

# 6. Dataset

## PAP_NER

Use the PAP_NER dataset introduced in the original paper.

Dataset characteristics:

- Vietnamese administrative documents
- 162,801 sentences
- 205,807 annotated entities
- 5 entity types

Entity types:

| Label | Description |
|---|---|
| CQ | Administrative / Government Agency |
| VBPL | Legal Document |
| ĐT | Object / Subject |
| NG | Date / Time |
| SL | Quantity |

The dataset split and preprocessing procedure should follow the original PAP_NER paper as closely as possible.

---

# 7. Baseline

## Original Baseline

The main baseline is:

> PhoBERT + CRF

Reported result in the original PAP_NER paper:

> Micro-F1 = 97.95%

The first experiment should reproduce this result as closely as possible.

### Baseline experiment

```text
PAP_NER
   |
   v
PhoBERT
   |
   v
CRF
   |
   v
BIO/BIOES prediction
   |
   v
Precision / Recall / Micro-F1
```

# 8. Proposed Benchmark

The proposed study does not introduce a new NER architecture.

Instead, the study replaces the pretrained language model while keeping the downstream architecture and evaluation protocol as consistent as possible.

```text
                    PAP_NER
                       |
        +--------------+--------------+
        |              |              |
        v              v              v
    PhoBERT          Model A         Model B
        |              |              |
        v              v              v
       CRF            CRF            CRF
        |              |              |
        +--------------+--------------+
                       |
                    Evaluation
```

The main independent variable is:

> **Pre-trained Language Model**

The CRF layer, dataset, task, and evaluation metrics should remain unchanged whenever possible.


# 9. Model Selection

Select 2–3 recent pretrained language models.

## Selection Criteria

1. Supports Vietnamese or multilingual Vietnamese text.
2. Released more recently than PhoBERT.
3. Has publicly available pretrained weights.
4. Has sufficient documentation or existing implementations.
5. Can be fine-tuned within the available GPU resources.
6. Has demonstrated strong performance on Vietnamese NLP tasks.
7. Has not already been extensively benchmarked on PAP_NER.

Candidate models should be finalized after reviewing related work from 2024–2026.


# 10. Experimental Setup

Keep the experimental environment consistent across models.

## 10.1. Hardware

Record:

- GPU
- VRAM
- CPU
- RAM
- Storage

## 10.2. Software

Record:

- Python version
- PyTorch version
- Transformers version
- CUDA version
- Operating system

## 10.3. Training Configuration

Keep the following settings identical where possible:

- Train / validation / test split
- Batch size
- Number of epochs
- Learning rate
- Maximum sequence length
- Optimizer
- Weight decay
- Random seed

If different models require different settings, clearly document the differences.


# 11. Evaluation Metrics

The primary metric should be:

> **Micro-F1**

Also report:

- Precision
- Recall
- F1-score

Evaluate both overall performance and entity-level performance.

## 11.1. Overall Performance

```text
Precision
Recall
Micro-F1
```

## 11.2. Entity-level Performance

```text
CQ
VBPL
ĐT
NG
SL
```

This allows the study to determine whether a model improves only overall performance or also improves specific entity categories.


# 12. Main Experiment

Run the following experiments:

| Experiment | Model | CRF |
|---|---|---|
| Baseline | PhoBERT | Yes |
| Experiment 1 | Model A | Yes |
| Experiment 2 | Model B | Yes |
| Experiment 3 | Model C | Yes |

The primary objective is to determine whether replacing PhoBERT with a more recent pretrained language model improves the benchmark performance.

The baseline result reported by PAP_NER is:

> **PhoBERT + CRF: 97.95 Micro-F1**


# 13. Ablation Study

Perform a limited ablation study if computational resources permit.

For example:

| Model | CRF | F1 |
|---|---:|---:|
| PhoBERT | No | ... |
| PhoBERT | Yes | 97.95 |
| Model A | No | ... |
| Model A | Yes | ... |

The purpose is to determine whether the performance improvement comes from the pretrained language model itself or from the CRF layer.

The ablation study should remain small because of the limited project duration.


# 14. Entity-level Analysis

Compare the models across the five entity types.

| Entity | PhoBERT-CRF | Model A-CRF | Model B-CRF |
|---|---:|---:|---:|
| CQ | ... | ... | ... |
| VBPL | ... | ... | ... |
| ĐT | ... | ... | ... |
| NG | ... | ... | ... |
| SL | ... | ... | ... |

Analyze:

- Which entity types are easiest?
- Which entity types are hardest?
- Which model performs best for each entity?
- Does the overall improvement come from one particular entity type?

This analysis is important because an improvement in overall Micro-F1 may be caused mainly by improvement in one entity category.


# 15. Error Analysis

Perform qualitative error analysis on representative examples.

Focus on the following error categories:

## 15.1. Entity Boundary Errors

Example:

```text
Expected:
[Ủy ban nhân dân xã Đồng Văn]

Predicted:
[Ủy ban nhân dân]
```

Analyze whether newer models are better at identifying the complete entity span.

## 15.2. Entity Type Errors

Example:

```text
Expected:
[Công văn số 123] → VBPL

Predicted:
[Công văn số 123] → CQ
```

The model identifies the correct span but assigns an incorrect entity type.

## 15.3. Long Entity Errors

Investigate whether models have difficulty identifying long administrative entities.

## 15.4. Ambiguous Entities

Some words or phrases may have different meanings depending on the context.

For example, an administrative term may refer to an organization in one sentence but another entity type in a different context.

## 15.5. Rare Entities

Analyze errors involving entity types or patterns with relatively few training examples.

The goal is not only to determine which model has the highest F1-score, but also to understand why the models perform differently.


# 16. Computational Efficiency

Accuracy should not be the only comparison criterion.

Record:

- Number of parameters
- Model size
- Training time
- Inference time
- Peak GPU memory usage

Example:

| Model | Micro-F1 | Parameters | GPU Memory | Inference Time |
|---|---:|---:|---:|---:|
| PhoBERT-CRF | 97.95 | ... | ... | ... |
| Model A-CRF | ... | ... | ... | ... |
| Model B-CRF | ... | ... | ... | ... |

This provides an accuracy–efficiency comparison.

For example, if Model A improves F1 by only 0.1% but requires substantially more computational resources, its practical advantage may be limited.


# 17. Statistical Analysis

If computational resources permit, run each experiment with multiple random seeds.

For example:

```text
Seed 42
Seed 123
Seed 2024
```

Report:

```text
Mean ± Standard Deviation
```

Example:

```text
PhoBERT-CRF: 97.95 ± 0.05
Model A-CRF: 98.12 ± 0.08
```

This provides stronger evidence that the observed difference is not simply caused by random initialization.

If the difference is sufficiently large, a statistical significance test can also be considered.


# 18. Expected Research Contribution

The expected contribution of this study is primarily empirical rather than architectural.

## Contribution 1: Benchmarking

Provide a systematic benchmark of recent pretrained language models for Vietnamese Administrative NER.

## Contribution 2: Reproducible Comparison

Compare recent models against the PhoBERT-CRF baseline from PAP_NER under a consistent experimental setting.

## Contribution 3: Entity-level Analysis

Analyze model performance across the five administrative entity categories:

```text
CQ
VBPL
ĐT
NG
SL
```

## Contribution 4: Error Analysis

Identify the main types of errors made by different pretrained language models.

## Contribution 5: Accuracy–Efficiency Analysis

Compare not only accuracy but also computational cost.

Therefore, the contribution can be summarized as:

> **An empirical evaluation and analysis of recent pretrained language models for Vietnamese Administrative Named Entity Recognition.**

The research does not claim to introduce a novel NER architecture.


# 19. Expected Results

There are two possible major outcomes.

## Case 1: A New Model Outperforms PhoBERT

For example:

```text
PhoBERT-CRF: 97.95
Model A-CRF: 98.20
```

The study can conclude that the newer pretrained language model provides better representations for Vietnamese administrative text and improves NER performance.

Further analysis should determine:

- Which entity types improved?
- Which error types were reduced?
- Whether the improvement is statistically significant.
- Whether the improvement justifies the additional computational cost.


## Case 2: A New Model Does Not Outperform PhoBERT

For example:

```text
PhoBERT-CRF: 97.95
Model A-CRF: 97.80
```

This does not make the research invalid.

Instead, the study can investigate possible explanations:

- Domain mismatch
- Vietnamese-specific pretraining advantages
- Model size
- Entity-type differences
- Long-context behavior
- Training stability
- Differences in tokenization
- Computational constraints

This can lead to an important research finding:

> A newer or larger pretrained language model does not necessarily outperform a Vietnamese-specific pretrained model on a specialized administrative NER domain.

Therefore, the research does not need to guarantee a new SOTA result.

The primary objective is:

> **To empirically evaluate whether recent pretrained language models provide measurable advantages over PhoBERT for Vietnamese Administrative NER, and to analyze the reasons behind the observed performance differences.**


# 20. Research Questions

The study can formulate the following research questions.

## RQ1

> Can recent pretrained language models outperform the PhoBERT-CRF baseline on the PAP_NER dataset?

## RQ2

> How does the choice of pretrained language model affect NER performance across different Vietnamese administrative entity types?

## RQ3

> What types of errors are reduced or introduced when replacing PhoBERT with a more recent pretrained language model?

## RQ4

> What is the trade-off between NER performance and computational efficiency among the evaluated models?


# 21. Research Hypotheses

## Null Hypothesis (H0)

> Recent pretrained language models do not provide a statistically significant improvement over PhoBERT-CRF for Vietnamese Administrative NER.

## Alternative Hypothesis (H1)

> At least one recent pretrained language model provides a statistically significant improvement over PhoBERT-CRF for Vietnamese Administrative NER.

A more specific hypothesis can also be considered:

> H1: A recent pretrained language model with stronger multilingual or Vietnamese contextual representations will achieve higher Micro-F1 than PhoBERT-CRF on PAP_NER.


# 22. Research Scope

To ensure that the project can be completed within approximately 3–4 weeks, the scope should be intentionally limited.

## Included

- PAP_NER dataset
- Vietnamese Administrative NER
- PhoBERT-CRF baseline
- 2–3 recent pretrained language models
- Standard fine-tuning
- Micro-F1, Precision, Recall
- Entity-level evaluation
- Error analysis
- Computational efficiency analysis
- Limited statistical analysis

## Excluded

The study will not focus on:

- Creating a new dataset
- Manual annotation of a new corpus
- Developing a completely new NER architecture
- Retrieval-Augmented Generation (RAG)
- Knowledge Graph construction
- Large-scale instruction tuning
- Building a new NLP framework
- Deploying a production system

This limitation is important because the objective is to perform a rigorous empirical comparison rather than maximize architectural novelty.


# 23. Proposed Research Workflow

The complete workflow can be summarized as follows:

```text
              Literature Review
                     |
                     v
               PAP_NER Study
                     |
                     v
            Dataset Preparation
                     |
                     v
          Reproduce PhoBERT-CRF
                     |
                     v
             Baseline Validation
                     |
                     v
        Select Recent PLM Candidates
                     |
          +----------+----------+
          |          |          |
          v          v          v
       Model A    Model B    Model C
          |          |          |
          v          v          v
        + CRF      + CRF      + CRF
          |          |          |
          +----------+----------+
                     |
                     v
              Model Evaluation
                     |
          +----------+----------+
          |          |          |
          v          v          v
       Overall   Entity-level  Efficiency
       Metrics     Metrics      Metrics
          |          |          |
          +----------+----------+
                     |
                     v
               Error Analysis
                     |
                     v
            Statistical Analysis
                     |
                     v
             Result Discussion
                     |
                     v
                Conclusion
```


# 24. Proposed 4-Week Schedule

Because all four members have other coursework and full-time jobs, the experimental scope should be controlled carefully.

## Week 1 — Baseline and Dataset

### Member 1
- Study PAP_NER
- Prepare dataset
- Understand BIO labels
- Prepare preprocessing pipeline

### Member 2
- Reproduce PhoBERT-CRF
- Verify the implementation
- Establish baseline performance

### Member 3
- Research recent Vietnamese/multilingual PLMs
- Identify candidate models

### Member 4
- Study evaluation methodology
- Prepare evaluation scripts
- Prepare error-analysis framework

### Deliverable

```text
Dataset ready
PhoBERT-CRF running
Baseline result obtained
2–3 candidate PLMs selected
```


## Week 2 — Main Experiments

### Member 1
- Support preprocessing and data pipeline

### Member 2
- Train Model A

### Member 3
- Train Model B

### Member 4
- Train Model C or support evaluation

### Deliverable

```text
PhoBERT-CRF
Model A-CRF
Model B-CRF
Model C-CRF
```

with initial results.


## Week 3 — Analysis

Perform:

- Final training runs
- Multiple seeds if possible
- Precision / Recall / F1
- Entity-level F1
- Error analysis
- Computational efficiency measurement
- Statistical analysis

### Deliverable

Complete experimental results.


## Week 4 — Paper and Presentation

### Tasks

- Introduction
- Related Work
- Methodology
- Experimental Setup
- Results
- Discussion
- Error Analysis
- Conclusion
- Figures
- Tables
- Presentation slides

### Final Deliverables

```text
Research Paper
Source Code
Experimental Results
Presentation Slides
```


# 25. Division of Work for Four Members

A practical division could be:

| Member | Main Responsibility |
|---|---|
| Member 1 | Dataset + preprocessing + baseline |
| Member 2 | Model A + experiments |
| Member 3 | Model B/C + experiments |
| Member 4 | Evaluation + error analysis + statistics |

However, all members should participate in:

- Literature review
- Result discussion
- Paper writing
- Presentation preparation


# 26. Final Research Design

The final research design can be summarized as:

```text
Research Topic:
Benchmarking Recent Pre-trained Language Models
for Vietnamese Administrative Named Entity Recognition

                PAP_NER Dataset
                       |
        +--------------+--------------+
        |              |              |
        v              v              v
     PhoBERT        Model A         Model B
        |              |              |
        v              v              v
       CRF            CRF            CRF
        |              |              |
        +--------------+--------------+
                       |
                       v
                  Evaluation
                       |
        +--------------+--------------+
        |              |              |
        v              v              v
    Overall       Entity-level    Efficiency
    Metrics         Metrics        Metrics
        |              |              |
        +--------------+--------------+
                       |
                       v
                Error Analysis
                       |
                       v
              Statistical Analysis
                       |
                       v
                  Conclusion
```

The key research question is:

> **Does replacing PhoBERT with a recent pretrained language model improve Vietnamese Administrative Named Entity Recognition when the downstream CRF architecture and evaluation setting are kept consistent?**

The key methodological principle is:

> **Change one major factor — the pretrained language model — while keeping the rest of the experimental setup as controlled as possible.**

This makes the study feasible within the limited 3–4 week timeframe while still providing a clear experimental research question, measurable hypotheses, reproducible experiments, and meaningful analysis.