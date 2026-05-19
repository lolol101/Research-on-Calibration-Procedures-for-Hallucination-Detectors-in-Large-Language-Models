# Hallucination_Research

## Project Title

**Research on Calibration Procedures for Hallucination Detectors in Large Language Models**

## Description

This project focuses on improving the reliability of hallucination detection in Large Language Models (LLMs). Hallucinations refer to outputs that are fluent and plausible but factually incorrect. The goal of this research is to develop a novel calibration method that produces confidence scores aligned with the true correctness of model responses.

Unlike existing approaches that rely mainly on final output probabilities, this work leverages internal model signals, including attention mechanisms and intermediate reasoning steps (Chain of Thought). The proposed method uses entropy-based metrics to extract uncertainty information and improve calibration quality.

## Key Ideas

* Use internal attention states instead of only final output logits
* Incorporate reasoning traces (Chain of Thought) for richer signals
* Apply entropy-based estimation for uncertainty measurement
* Select “hallucination-aware” attention heads based on separation ability
* Aggregate multiple signals into a calibrated confidence score

## Methodology

The approach consists of two main stages:

1. **Attention Head Selection**
   Identify attention heads that best distinguish between factual and hallucinated outputs using entropy-based separation metrics.

2. **Confidence Aggregation and Calibration**
   Combine signals from selected attention heads and output layers into a single confidence score. Apply calibration techniques such as temperature scaling, beta calibration and MLP head to align scores with true probabilities.

## Models

Experiments are conducted on open-source LLMs with accessible internal states:

* Qwen2.5-7B-Instruct
* Llama3-8B

## Datasets

* MMLU-PRO
* RACE

These datasets provide labeled data necessary for evaluating hallucination detection and calibration.

## Evaluation

The method is evaluated using:

* **Detection Quality:** ROC-AUC score for distinguishing hallucinated vs factual outputs
* **Calibration Quality:** Expected Calibration Error (ECE) and its variants

## Expected Outcomes

* Improved calibration of confidence scores
* Better detection of hallucinated outputs
* Enhanced understanding of internal model signals for uncertainty estimation

## References

The project builds upon prior work in calibration, entropy-based uncertainty estimation, and attention-based hallucination detection.
