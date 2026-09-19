# RCAFN-CVD-Risk-Stratification

## Explainable Retinal-Clinical Adaptive Fusion Network (RCAFN) for Early Cardiovascular Disease Risk Stratification

<p align="center">

**An Explainable Multimodal Deep Learning Framework for CIMT-Based Cardiovascular Risk Stratification Using Retinal Fundus Images and Clinical Risk Factors**

</p>

---

## 📌 Overview

Cardiovascular disease (CVD) is one of the major causes of illness and death worldwide. Traditional cardiovascular risk assessment primarily depends on clinical variables such as age, gender, blood pressure, lipid levels, diabetes status, and other risk factors.

Recent advances in deep learning have demonstrated that **retinal fundus photographs contain vascular information associated with systemic cardiovascular health**. At the same time, clinical variables provide complementary patient-level information.

This project proposes **RCAFN (Retinal-Clinical Adaptive Fusion Network)**, an explainable multimodal deep learning framework that combines:

* 🩺 Retinal fundus images
* 👤 Clinical information such as age and gender
* 🧠 Deep image representations extracted using ResNet-50
* 🔀 Patient-adaptive gated multimodal fusion
* 🔬 CIMT-based cardiovascular/atherosclerotic risk stratification
* 👁️ Grad-CAM-based visual explanations
* 🩸 Quantitative evaluation of whether model explanations are grounded in retinal vascular structures

The central research question is:

> **Does patient-adaptive gated fusion of retinal and clinical representations provide a meaningful improvement over conventional multimodal feature concatenation for CIMT-based cardiovascular risk stratification?**

The project is designed as a **research and publication-oriented study**, rather than simply reproducing an existing multimodal model.

---

# 🎯 Research Objectives

The major objectives of this project are:

1. Develop a deep learning model for cardiovascular risk stratification using retinal fundus images.
2. Incorporate clinical information such as age and gender.
3. Establish strong unimodal and conventional multimodal baselines.
4. Develop a **patient-adaptive gated fusion mechanism**.
5. Compare adaptive fusion against conventional feature concatenation.
6. Investigate whether the contribution of retinal and clinical modalities should vary between patients.
7. Generate Grad-CAM visual explanations.
8. Quantitatively evaluate whether model saliency overlaps with retinal vascular structures.
9. Perform controlled ablation experiments.
10. Evaluate the models using clinically relevant classification metrics and statistical analysis.
11. Develop a Streamlit-based research prototype.
12. Prepare reproducible experimental results suitable for research publication.

---

# 💡 Key Research Contributions

## Primary Contribution

### Patient-Adaptive Gated Multimodal Fusion

Instead of simply concatenating retinal and clinical features, RCAFN learns a **sample-specific gating mechanism**.

The model learns how much importance should be assigned to:

* Retinal information
* Clinical information

for each individual patient.

This allows the model to adapt the relative contribution of the two modalities on a patient-by-patient basis.

---

## Supporting Contribution

### Quantitative Vessel-Grounded Explainability

Grad-CAM will be used to generate visual explanations of the retinal model.

Instead of only presenting example heatmaps, the project evaluates whether the important regions identified by the model correspond to **retinal vascular structures**.

A predefined quantitative vessel-grounded saliency/overlap metric will be used.

This provides a more rigorous evaluation of explainability.

---

# 🏗️ Proposed Architecture

```text
                    RETINAL MODALITY
                         │
                         ▼
                 Fundus Image(s)
                         │
                         ▼
                     ResNet-50
                         │
                         ▼
              Retinal Representation
                         │
                         │
                         ├──────────────┐
                         │              │
                         ▼              ▼
                    Projection     Clinical Data
                         │          Age + Gender
                         │              │
                         │              ▼
                         │       Clinical Encoder
                         │              │
                         │              ▼
                         │      Clinical Representation
                         │              │
                         └──────┬───────┘
                                │
                                ▼
                       Learned Sigmoid Gate
                                │
                                ▼
                     Adaptive Multimodal Fusion
                                │
                                ▼
                       Classification Head
                                │
                                ▼
                  CIMT-Based Risk Stratification
                                │
                                ▼
                         Risk Prediction


              EXPLAINABILITY PIPELINE
                         │
                         ▼
                      Grad-CAM
                         │
                         ▼
                 Saliency / Heatmap
                         │
                         ▼
               Retinal Vessel Mask
                         │
                         ▼
          Quantitative Vessel-Grounded
               Saliency Evaluation
```

The proposed architecture consists of a retinal encoder, clinical encoder, common embedding space, learned sigmoid gate, adaptive fusion layer, and classification head.

---

# 🔬 Methodology

## 1. Retinal Image Encoder

Retinal fundus photographs are processed using **ResNet-50**.

The network extracts high-level retinal representations containing information from structures such as:

* Retinal blood vessels
* Optic disc
* Macular region
* Background retinal characteristics

The resulting feature vector represents the retinal modality.

---

## 2. Clinical Encoder

Clinical variables currently considered include:

* Age
* Gender

These variables are passed through a clinical encoder to obtain a numerical clinical representation.

Age will be appropriately normalized, while gender will be encoded numerically.

---

## 3. Common Embedding Space

The retinal and clinical representations may have different dimensionalities.

Therefore, both representations are projected into a **common embedding dimension** before fusion.

```text
Retinal Features
      │
      ▼
Projection Layer
      │
      ▼
Common Dimension

Clinical Features
      │
      ▼
Projection Layer
      │
      ▼
Common Dimension
```

---

# 🔀 Adaptive Gated Fusion

The central component of RCAFN is the **learned sigmoid gating mechanism**.

Instead of:

```text
Retinal Features + Clinical Features
              │
              ▼
       Concatenation
              │
              ▼
         Classifier
```

RCAFN performs:

```text
Retinal Representation
          │
          ├──────────────┐
          │              │
          ▼              │
     Projection          │
          │              │
          │         Learned Gate
          │              │
          │              ▼
          │        Sample-Specific
          │           Weight
          │
          └──────────────┐
                         │
Clinical Representation │
          │              │
          ▼              │
     Projection          │
                         │
                         ▼
                 Adaptive Fusion
                         │
                         ▼
                    Classifier
```

The gate produces learned weights that control the contribution of the modalities for each patient.

A sigmoid-based gate ensures that the learned gating values can be interpreted as modality contribution weights.

> **Important terminology:** The proposed mechanism should be called **"gated adaptive fusion"** unless an actual attention mechanism is implemented. It should not be described as attention merely because it assigns weights.

---

# 👁️ Explainability

## Grad-CAM

Gradient-weighted Class Activation Mapping (**Grad-CAM**) will be used to visualize the retinal regions contributing to the model prediction.

The pipeline is:

```text
Fundus Image
     │
     ▼
ResNet-50
     │
     ▼
Prediction
     │
     ▼
Grad-CAM
     │
     ▼
Saliency Heatmap
```

The heatmap can help determine whether the model is relying on visually meaningful retinal regions.

---

# 🩸 Vessel-Grounded Explainability

A major supporting component of the project is the quantitative evaluation of Grad-CAM explanations.

A retinal vessel segmentation mask will be generated or obtained.

The Grad-CAM saliency map will then be compared with the vessel mask.

```text
                 Fundus Image
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
      Grad-CAM              Vessel Segmentation
          │                       │
          ▼                       ▼
   Saliency Map              Vessel Mask
          │                       │
          └───────────┬───────────┘
                      ▼
             Overlap Analysis
                      │
                      ▼
      Vessel-Grounded Saliency Metric
```

The exact metric will be predefined and applied consistently across the major models.

The goal is **not** to claim that Grad-CAM itself is novel.

Instead, the research contribution is the **quantitative evaluation of whether model explanations are grounded in retinal vascular structures**.

---

# 📊 Experimental Design

A major requirement of the project is to compare RCAFN against appropriate baselines.

## Core Experimental Matrix

| Model               | Retinal Image | Clinical Data    | Purpose                             |
| ------------------- | ------------- | ---------------- | ----------------------------------- |
| Clinical-only       | ❌             | Age + Gender     | Clinical baseline                   |
| Image-only          | ResNet-50     | ❌                | Primary image baseline              |
| Conventional Fusion | ResNet-50     | Age + Gender     | Existing/simple multimodal baseline |
| **RCAFN**           | **ResNet-50** | **Age + Gender** | **Proposed adaptive/gated fusion**  |

The **key research comparison is Conventional Fusion vs RCAFN**.

---

# 🧪 Experimental Phases

## Phase 0 — Literature and Project Freeze

Before implementation:

* Study relevant literature.
* Create a literature comparison table.
* Identify existing multimodal approaches.
* Freeze the research question.
* Freeze the primary and supporting contributions.

This prevents the project from becoming a simple reimplementation.

---

## Phase 1 — Dataset Acquisition and Understanding

The project will use the authorized **China-Fundus-CIMT dataset**.

The dataset should be inspected for:

* Patient IDs
* Left/right eye information
* Fundus images
* CIMT measurements
* Age
* Gender
* Missing values
* Class balance
* Image quality

### Data Privacy

> **Medical data and sensitive patient information must NOT be uploaded to GitHub.**

Only dataset access instructions and expected directory structure should be maintained in the repository.

---

# 🔐 Phase 2 — Patient-Level Data Splitting

This is a critical methodological requirement.

Train, validation and test sets must be created **by patient ID**, not by individual image.

If a patient has:

```text
Patient 001
 ├── Left Eye
 └── Right Eye
```

both images must belong to the same split.

### Correct

```text
Train
 ├── Patient 001
 ├── Patient 002
 └── Patient 003

Validation
 ├── Patient 004

Test
 ├── Patient 005
```

### Incorrect

```text
Train
 └── Patient 001 - Left Eye

Test
 └── Patient 001 - Right Eye
```

The second approach creates **patient-level data leakage**.

The final split and class distribution should be recorded for reproducibility.

---

# 🖼️ Phase 3 — Image and Clinical Preprocessing

## Retinal Images

The preprocessing pipeline will include:

* Image resizing
* Normalization
* Quality inspection
* Appropriate training augmentation

CLAHE may be evaluated experimentally rather than automatically assuming that it improves performance.

Augmentation must be applied only to training data.

---

## Clinical Variables

### Age

Age will be:

* Validated
* Normalized
* Converted into a numerical representation

### Gender

Gender will be encoded into a numerical representation.

Missing clinical values must be handled consistently.

---

# 🧠 Phase 4 — Baseline Models

Three baselines will be implemented before RCAFN.

### Baseline 1 — Clinical Only

```text
Age + Gender
      │
      ▼
     MLP
      │
      ▼
Prediction
```

Purpose:

Determine how much predictive information is available from clinical variables alone.

---

### Baseline 2 — Image Only

```text
Fundus Image
     │
     ▼
 ResNet-50
     │
     ▼
Prediction
```

Purpose:

Measure the predictive capability of retinal images independently.

---

### Baseline 3 — Conventional Multimodal Fusion

```text
             Fundus
               │
               ▼
           ResNet-50
               │
               ▼
        Image Embedding
               │
               ├─────────────┐
                             │
Age + Gender ──► Encoder ────┤
                             ▼
                       Concatenation
                             │
                             ▼
                         Classifier
                             │
                             ▼
                        Prediction
```

This is the most important baseline because RCAFN's proposed contribution is an improvement over conventional multimodal fusion.

---

# 🚀 Phase 5 — RCAFN Implementation

The proposed model will contain:

1. Image encoder
2. Clinical encoder
3. Image projection layer
4. Clinical projection layer
5. Learned sigmoid gate
6. Adaptive fusion mechanism
7. Classification head

Training will record the learned gate values for further analysis.

---

# 👀 Phase 6 — Bilateral Analysis

If the dataset quality permits, bilateral fundus information will be investigated.

Possible configurations:

* Left eye only
* Right eye only
* Bilateral images

The bilateral experiment will initially be treated as a supporting experiment rather than the main contribution.

Most importantly:

> Both eyes belonging to the same patient must always remain in the same dataset split.

---

# 🔬 Phase 7 — Explainability Experiments

For major models:

* Generate Grad-CAM maps.
* Obtain/create retinal vessel masks.
* Calculate the predefined vessel-grounded saliency metric.
* Compare explanation behavior between models.

Models may include:

* Image-only
* Conventional fusion
* RCAFN

The evaluation should include quantitative analysis rather than only visual examples.

---

# 🧪 Phase 8 — Ablation Studies

Ablation studies will determine whether each proposed component actually contributes to performance.

## Main Ablations

### A. Image-only vs Conventional Fusion vs RCAFN

```text
Image-only
    ↓
Conventional Fusion
    ↓
RCAFN
```

### B. Gate Ablation

Remove or replace the adaptive gate.

Compare:

```text
Without Gate
      vs
With Gate
```

This isolates the contribution of the proposed gating mechanism.

### Optional Ablations

* CLAHE vs no CLAHE
* Single-eye vs bilateral input
* Other preprocessing variations

Ablations should be performed in a controlled manner.

---

# 📈 Phase 9 — Statistical Evaluation

## Primary Metric

### AUROC

The primary evaluation metric will be:

**Area Under the Receiver Operating Characteristic Curve (AUROC)**

---

## Secondary Metrics

The following metrics will also be reported:

* AUPRC
* Sensitivity
* Specificity
* Precision
* F1-score
* Accuracy
* Confusion matrix

Confidence intervals should be reported for major metrics where feasible.

A suitable paired statistical comparison should be used for the primary model-vs-baseline comparison.

Importantly:

> Performance improvement should not be claimed unless it is supported by the experimental and statistical evidence.

---

# 🛡️ Phase 10 — Robustness Testing

This phase is optional and should be performed after the core model is stable.

Potential image perturbations include:

* Brightness changes
* Contrast changes
* Blur
* Noise
* Compression artifacts

The objective is to measure:

1. Performance degradation
2. Prediction stability
3. Model robustness

---

# 🌐 Phase 11 — Streamlit Research Prototype

After the research model is frozen, a Streamlit prototype can be developed.

## Input

The prototype will accept:

* Fundus image(s)
* Age
* Gender

## Output

The application can display:

* Predicted risk category
* Model confidence/probability
* Grad-CAM explanation
* Relevant retinal regions

### Important

The application must clearly state:

> **This is a research prototype and is not a diagnostic medical tool.**

The Streamlit layer should be developed **after** the research model and experiments are finalized.

---

# 📚 Dataset

## China-Fundus-CIMT

The project is designed around the **China-Fundus-CIMT** dataset.

The dataset paper reports:

* 2,903 patients
* Bilateral fundus images
* CIMT measurements
* Age
* Gender

The dataset is particularly relevant to this project because it combines retinal fundus information with cardiovascular-related CIMT measurements.

### Data Policy

The actual medical dataset must **not** be committed to this repository.

The `data/README.md` file should contain only:

* Dataset description
* Authorized access instructions
* Expected folder structure
* Preprocessing requirements

---

# 📁 Project Structure

```text
RCAFN-CVD-Risk-Stratification/
│
├── README.md
│
├── docs/
│   ├── project-abstract.pdf
│   ├── literature-review.md
│   ├── methodology.md
│   └── references.md
│
├── data/
│   └── README.md
│
├── notebooks/
│   ├── 01_dataset_analysis.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_training.ipynb
│   ├── 04_conventional_fusion.ipynb
│   ├── 05_rcafn_training.ipynb
│   └── 06_explainability.ipynb
│
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── models/
│   ├── explainability/
│   └── evaluation/
│
├── results/
│
├── app/
│
├── requirements.txt
│
└── .gitignore
```

This structure separates documentation, data instructions, experiments, source code, results and application code.

---

# 👥 Team Responsibilities

| Team Member            | Primary Responsibility                        | Secondary Responsibility                       |
| ---------------------- | --------------------------------------------- | ---------------------------------------------- |
| **Harsha – Team Lead** | RCAFN architecture, gated fusion, integration | GitHub, documentation, experiment coordination |
| **Rohit**              | Dataset, metadata, patient-level splitting    | Preprocessing/data pipeline                    |
| **Praneeth**           | Baseline models and training                  | Evaluation scripts/experiment tracking         |
| **Akhil**              | Grad-CAM and vessel-grounded explainability   | Visualization/statistical analysis             |

These responsibilities are designed to divide the research implementation into complementary workstreams.

---

# 🗓️ Project Milestones

| Milestone | Goal                               | Evidence                                                   |
| --------- | ---------------------------------- | ---------------------------------------------------------- |
| **M1**    | Literature + dataset understanding | Literature matrix, dataset notes, frozen research question |
| **M2**    | Data pipeline                      | Patient-level split and reproducible loaders               |
| **M3**    | Baselines                          | Clinical-only, image-only and conventional fusion results  |
| **M4**    | RCAFN                              | Working gated fusion model                                 |
| **M5**    | Core experiments                   | Baseline vs RCAFN + ablations                              |
| **M6**    | Explainability                     | Grad-CAM + quantitative vessel-grounded evaluation         |
| **M7**    | Statistics                         | Confidence intervals and main comparisons                  |
| **M8**    | Demo                               | Streamlit research prototype                               |
| **M9**    | Paper package                      | Figures, tables, methods, discussion and limitations       |

---

# 🔍 Research Comparison

The project is based on and compared with recent work in retinal cardiovascular risk prediction.

## 1. Lee et al. (2023) — Base Paper

**Multimodal deep learning of fundus abnormalities and traditional risk factors for cardiovascular risk prediction**

This work establishes the use of multimodal retinal fundus and clinical information for cardiovascular risk prediction.

* DOI: https://doi.org/10.1038/s41746-023-00748-4
* Article: https://www.nature.com/articles/s41746-023-00748-4

---

## 2. Li et al. (2024) — Scoping Review

**Prediction of cardiovascular markers and diseases using retinal fundus images and deep learning: a systematic scoping review**

This paper maps research in retinal fundus-based cardiovascular prediction and highlights validation and clinical-comparison gaps.

* DOI: https://doi.org/10.1093/ehjdh/ztae068
* Article: https://academic.oup.com/ehjdh/article/5/6/660/7754720

---

## 3. Gong et al. (2024) — CIMT and Grad-CAM

**A Siamese ResNeXt network for predicting carotid intimal thickness of patients with T2DM from fundus images**

This work is directly relevant to:

* Fundus → CIMT prediction

* ResNet/ResNeXt architectures

* Age embedding

* Grad-CAM explainability

* DOI: https://doi.org/10.3389/fendo.2024.1364519

* Article: https://www.frontiersin.org/journals/endocrinology/articles/10.3389/fendo.2024.1364519/full

---

## 4. Guo et al. (2025) — China-Fundus-CIMT

**High-resolution fundus images for ophthalmomics and early cardiovascular disease prediction**

This is the closest dataset-related work.

It introduces China-Fundus-CIMT containing:

* 2,903 patients
* Bilateral fundus images
* CIMT
* Age
* Gender

The work also reports multimodal gains.

* DOI: https://doi.org/10.1038/s41597-025-04930-z
* Article: https://www.nature.com/articles/s41597-025-04930-z

---

## 5. Li et al. (2025) — Multimodal CAD

**Noninvasive Coronary Artery Disease Detection Using Retinal Images: A Multimodal Study**

This recent work demonstrates multimodal cardiovascular modelling using retinal and clinical information and includes cross-modal attention approaches.

* DOI: https://doi.org/10.1016/j.jacadv.2025.102341
* Article: https://www.jacc.org/doi/full/10.1016/j.jacadv.2025.102341

The comparison literature indicates that fundus + clinical data, CIMT prediction, CNN-based approaches and Grad-CAM are already established components. Therefore, the proposed research focuses on **adaptive fusion and quantitative explainability**, rather than claiming these individual components as novel.

---

# 🧩 Research Gap

Based on the literature reviewed for this project, several areas motivate RCAFN:

### Existing approaches commonly use:

* Retinal fundus images
* Clinical risk factors
* CNN-based feature extraction
* Multimodal feature fusion
* CIMT-related prediction
* Grad-CAM visualization

### RCAFN focuses on:

```text
Existing Multimodal Models
          │
          ▼
Simple / Conventional Fusion
          │
          │
          ▼
       RCAFN
          │
          ├── Patient-Adaptive Gated Fusion
          │
          └── Quantitative Vessel-Grounded
              Explainability
```

The main hypothesis is that **patient-adaptive weighting of modalities may provide a better multimodal representation than fixed concatenation**.

However, the project will only claim this if the experiments demonstrate statistically supported improvement.

---

# 🧪 Research Hypothesis

## Primary Hypothesis

> **Patient-adaptive gated fusion of retinal and clinical representations can improve CIMT-based cardiovascular risk stratification compared with conventional feature concatenation.**

## Secondary Hypothesis

> **Quantitative evaluation of Grad-CAM saliency against retinal vessel structures can provide stronger evidence about whether model explanations are biologically grounded.**

---

# 📊 Expected Results

The project will compare:

```text
Clinical-only
     │
     ▼
Image-only
     │
     ▼
Conventional Fusion
     │
     ▼
RCAFN
```

The final analysis will determine whether:

* Multimodal information improves over individual modalities.
* Conventional fusion improves over unimodal models.
* Adaptive gating improves over conventional fusion.
* The learned gate produces meaningful patient-level modality weighting.
* RCAFN explanations demonstrate stronger vessel-grounded behavior.
* Improvements are statistically supported.

**No performance improvement will be assumed in advance.**

---

# 📈 Result Reporting

Final experiments should report a table similar to:

| Model               |   AUROC |   AUPRC | Sensitivity | Specificity | Precision |      F1 | Accuracy |
| ------------------- | ------: | ------: | ----------: | ----------: | --------: | ------: | -------: |
| Clinical-only       |     TBD |     TBD |         TBD |         TBD |       TBD |     TBD |      TBD |
| Image-only          |     TBD |     TBD |         TBD |         TBD |       TBD |     TBD |      TBD |
| Conventional Fusion |     TBD |     TBD |         TBD |         TBD |       TBD |     TBD |      TBD |
| **RCAFN**           | **TBD** | **TBD** |     **TBD** |     **TBD** |   **TBD** | **TBD** |  **TBD** |

Confidence intervals and appropriate statistical comparisons should be added to the final results.

---

# 📌 Publication-Oriented Success Criteria

The project will be considered methodologically successful when:

* [x] The research question is clearly different from simple reproduction.
* [ ] Conventional fusion baseline is implemented fairly.
* [ ] RCAFN adaptive fusion is precisely defined.
* [ ] RCAFN's contribution is experimentally isolated.
* [ ] Patient-level leakage is prevented.
* [ ] Controlled ablation experiments are completed.
* [ ] Appropriate metrics are reported.
* [ ] Statistical comparisons are performed.
* [ ] Explainability is quantitatively evaluated.
* [ ] Limitations are documented.
* [ ] Negative results are reported honestly.
* [ ] Code and configurations are sufficiently reproducible.
* [ ] Final figures and tables are prepared.

Publication is **not guaranteed** by completing the implementation. Publication depends on the actual results, novelty relative to the latest literature, methodological rigor, writing quality, venue requirements and peer review.

---

# ⚠️ Important Terminology

To maintain scientific accuracy, this project uses the following terminology:

### Use:

> **CIMT-based cardiovascular/atherosclerotic risk stratification**

### Avoid:

> "Heart disease diagnosis"

The model is intended for **risk stratification/research**, not clinical diagnosis.

---

### Use:

> **Gated adaptive fusion**

unless a genuine attention mechanism is implemented.

### Avoid:

> Calling the gating mechanism "attention" without implementing an attention mechanism.

---

### Use:

> **Quantitative vascular-grounded explainability**

### Avoid:

> Claiming that Grad-CAM itself is novel.

Grad-CAM is an established explainability technique; the supporting contribution is its quantitative evaluation against retinal vascular structures.

---

# 🧠 Role of EyePACS / APTOS

If EyePACS or APTOS datasets are used, they should be treated as:

* Pretraining data
* Representation-learning data
* Auxiliary retinal image data

They should **not** automatically be treated as cardiovascular datasets unless a compatible cardiovascular label is available.

---

# 🔒 Data Privacy and GitHub Policy

This repository must **not contain**:

* Medical patient records
* Patient-identifiable information
* Restricted datasets
* Private clinical metadata
* Unauthorized fundus images

The repository should contain only:

```text
Code
Documentation
Configuration
Experiment scripts
Results that are safe to share
Dataset instructions
```

The actual medical dataset should remain in the authorized local/server environment.

---

# Reproducibility

To make the project reproducible, the repository should maintain:

* Fixed patient-level splits
* Dataset preprocessing configuration
* Model configuration
* Training configuration
* Random seeds
* Checkpoints
* Evaluation scripts
* Experiment logs
* Metric calculations
* Ablation configurations
* Final result tables

Example:

```text
results/
├── clinical_only/
├── image_only/
├── conventional_fusion/
├── rcafn/
├── ablations/
├── explainability/
└── statistical_analysis/
```

---

# Immediate Implementation Plan

The recommended order of implementation is:

### Step 1 — Repository Setup

Create the GitHub repository and add the team members as collaborators.

### Step 2 — Documentation

Add:

```text
docs/project-abstract.pdf
docs/literature-review.md
docs/methodology.md
docs/references.md
```

### Step 3 — Dataset

Obtain the authorized China-Fundus-CIMT dataset.

Inspect:

* Patient IDs
* Images
* CIMT
* Age
* Gender
* Missing values
* Class distribution
* Image quality

### Step 4 — Patient-Level Split

Finalize train/validation/test splits before model training.

### Step 5 — Baselines

Implement:

1. Clinical-only
2. Image-only
3. Conventional fusion

### Step 6 — RCAFN

Implement:

```text
ResNet-50
    +
Clinical Encoder
    +
Projection Layers
    +
Sigmoid Gate
    +
Adaptive Fusion
    +
Classifier
```

### Step 7 — Core Comparison

Compare:

```text
Conventional Fusion
        VS
       RCAFN
```

### Step 8 — Ablations

Test the contribution of the gate and other selected components.

### Step 9 — Explainability

Implement:

```text
Grad-CAM
   +
Vessel Mask
   +
Quantitative Overlap Metric
```

### Step 10 — Statistical Analysis

Calculate:

* AUROC
* AUPRC
* Sensitivity
* Specificity
* Precision
* F1
* Accuracy
* Confidence intervals
* Statistical comparison

### Step 11 — Demo

Develop the Streamlit prototype.

### Step 12 — Paper Preparation

Prepare:

* Methodology
* Results
* Tables
* Figures
* Ablation results
* Explainability results
* Statistical analysis
* Limitations
* Discussion
* Conclusion

The implementation order follows the project's publication-oriented plan.

---

# 🗺️ High-Level Workflow

```text
                    START
                      │
                      ▼
              Literature Review
                      │
                      ▼
            Dataset Understanding
                      │
                      ▼
          Patient-Level Data Split
                      │
                      ▼
               Preprocessing
                      │
                      ▼
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
    Clinical Baseline       Image Baseline
          │                       │
          └───────────┬───────────┘
                      ▼
             Conventional Fusion
                      │
                      ▼
                  RCAFN Model
                      │
                      ▼
              Adaptive Gated Fusion
                      │
                      ▼
            CIMT-Based Risk Output
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
      Ablation               Grad-CAM
       Studies                   │
          │                      ▼
          │              Retinal Vessel Mask
          │                      │
          │                      ▼
          │           Quantitative Explainability
          │                      │
          └───────────┬──────────┘
                      ▼
             Statistical Analysis
                      │
                      ▼
              Robustness Testing
                      │
                      ▼
             Streamlit Prototype
                      │
                      ▼
               Paper Preparation
                      │
                      ▼
                    END
```

---

# 🛠️ Planned Technology Stack

The implementation is expected to use:

* **Python**
* **PyTorch**
* **Torchvision**
* **OpenCV**
* **NumPy**
* **Pandas**
* **scikit-learn**
* **Matplotlib**
* **Grad-CAM tooling**
* **Streamlit**
* **Jupyter Notebook**
* **Git / GitHub**

The exact package versions should be frozen in:

```text
requirements.txt
```

after the implementation environment is finalized.

---

# 📜 Citation / References

If you use this repository or methodology in academic work, cite the relevant papers listed below.

### Lee et al. (2023)

**Multimodal deep learning of fundus abnormalities and traditional risk factors for cardiovascular risk prediction**

https://doi.org/10.1038/s41746-023-00748-4

https://www.nature.com/articles/s41746-023-00748-4

---

### Li et al. (2024)

**Prediction of cardiovascular markers and diseases using retinal fundus images and deep learning: a systematic scoping review**

https://doi.org/10.1093/ehjdh/ztae068

https://academic.oup.com/ehjdh/article/5/6/660/7754720

---

### Gong et al. (2024)

**A Siamese ResNeXt network for predicting carotid intimal thickness of patients with T2DM from fundus images**

https://doi.org/10.3389/fendo.2024.1364519

https://www.frontiersin.org/journals/endocrinology/articles/10.3389/fendo.2024.1364519/full

---

### Guo et al. (2025)

**High-resolution fundus images for ophthalmomics and early cardiovascular disease prediction**

https://doi.org/10.1038/s41597-025-04930-z

https://www.nature.com/articles/s41597-025-04930-z

---

### Li et al. (2025)

**Noninvasive Coronary Artery Disease Detection Using Retinal Images: A Multimodal Study**

https://doi.org/10.1016/j.jacadv.2025.102341

https://www.jacc.org/doi/full/10.1016/j.jacadv.2025.102341

---

# ⚠️ Disclaimer

This project is intended for **academic research and experimentation only**.

RCAFN is **not a medical device and should not be used for clinical diagnosis, treatment decisions, or patient management**.

The predictions produced by the research prototype should not be interpreted as a medical diagnosis.

---

# ⭐ Project Status

🚧 **Research & Development**

Current focus:

```text
Literature Review
      ↓
Dataset Preparation
      ↓
Baseline Implementation
      ↓
RCAFN Development
      ↓
Experimental Evaluation
      ↓
Explainability
      ↓
Statistical Analysis
      ↓
Research Prototype
      ↓
Publication Preparation
```

---

# 👨‍💻 Team

| Name         | Role                                         |
| ------------ | -------------------------------------------- |
| **Harsha**   | Team Lead — RCAFN Architecture & Integration |
| **Rohit**    | Dataset & Data Pipeline                      |
| **Praneeth** | Baselines & Model Training                   |
| **Akhil**    | Explainability & Statistical Analysis        |

---

# 📌 Key Research Question

> ### **Can patient-adaptive gated fusion of retinal and clinical representations improve CIMT-based cardiovascular risk stratification compared with conventional multimodal feature concatenation, while providing quantitatively more vessel-grounded explanations?**

---

## 🔬 Research Philosophy

The goal of this project is **not simply to build a model that produces a high accuracy score**.

The goal is to perform a controlled research study:

```text
Existing Methods
       ↓
Fair Baselines
       ↓
Proposed RCAFN
       ↓
Controlled Ablations
       ↓
Explainability Evaluation
       ↓
Statistical Validation
       ↓
Honest Conclusions
```

A positive result is valuable.

A negative result is also scientifically valuable if it demonstrates that adaptive fusion does not provide a meaningful improvement under controlled experimental conditions.

**The final conclusions will be driven by experimental evidence rather than assumptions.**
