# Camouflaged Object Detection using Heterogeneous Transformer-CNN Ensemble & Explainable AI

This project focuses on automated camouflaged object segmentation using a heterogeneous deep learning pipeline that combines Vision Transformers and Convolutional Neural Networks. The system produces pixel-wise segmentation masks for camouflaged objects, addressing the challenge of detecting targets that deliberately blend with their surroundings while emphasizing transparent and interpretable AI decision-making.

## Project Description

This project presents a robust camouflaged object detection system built using a **4-Model Heterogeneous Ensemble** (CamoNet) of complementary deep learning architectures. By integrating the global contextual reasoning of hierarchical Transformers with the local texture sensitivity of CNNs, the system achieves strong generalization across challenging camouflage scenarios. A shared decoder augmented with **frequency-domain analysis**, **edge-guided auxiliary supervision**, and **MC Dropout uncertainty estimation** further sharpens boundary delineation and model calibration. To promote trust in automated wildlife and military surveillance workflows, the project incorporates a comprehensive **Explainable AI (XAI)** framework (Grad-CAM, Integrated Gradients, and LIME), enabling clear visualization of the features influencing each prediction.

## Key Steps

### 1) Data Loading and Exploration

- Load the COD10K and CAMO camouflaged object datasets.
- Verify image quality and mask fidelity across diverse camouflage categories (aquatic, terrestrial, flying, amphibian).
- Visualize representative samples to understand variations in texture mimicry, background complexity, and object scale.
- Prepare stratified splits to support reliable cross-dataset evaluation.

### 2) Data Preprocessing

- **CamouflageAug**: Blends the foreground object with its surrounding background at random opacity, simulating stronger camouflage and forcing the model to learn from increasingly ambiguous object-background boundaries.
- **TextureMix & FrequencyJitter**: Perturbs the frequency spectrum of training images to improve robustness to texture-based camouflage.
- **CutMixSeg**: Applies region-level mixing to segmentation pairs for richer boundary supervision.
- **Normalization**: Uses ImageNet mean and standard deviation for compatibility with pretrained backbones.
- **Resizing**: Standardizes image resolution to 224×224 across all models.

### 3) Model Architecture

A diverse set of ImageNet-pretrained backbones is paired with a shared multi-branch decoder:

- **Swin-Base**: Hierarchical Transformer capturing multi-scale shifted-window attention for global context.
- **PoolFormer-S36**: Lightweight token-mixing Transformer providing efficient structural reasoning.
- **ResNet50**: Deep residual CNN for robust local texture and edge feature extraction.
- **EfficientNet-B4**: Compound-scaled CNN optimized for fine-grained detail at minimal parameter cost.

**Decoder Components:**

- **FPN + ASPP**: Multi-scale feature fusion with atrous spatial pyramid pooling for dense predictions.
- **CBAM (Channel-Spatial Attention)**: Recalibrates feature responses along both channel and spatial dimensions.
- **Frequency Branch**: FFT-based saliency head that captures complementary frequency-domain cues absent in spatial features.
- **Edge-Guided Auxiliary Head**: Boundary-supervised auxiliary decoder that sharpens object contour predictions.
- **Uncertainty Head (MC Dropout)**: Stochastic forward passes at inference produce calibrated pixel-wise uncertainty maps.

**Ensemble Strategy:**

- Extract probability maps from all four backbone-decoder models.
- Combine predictions using confidence-weighted averaging derived from per-pixel certainty scores.
- Exploit cross-family architectural diversity (Transformers vs. CNNs) to reduce individual model biases and improve boundary precision.

### 4) Training Strategy

- **Composite Loss**: Combines IoU loss, Dice loss, binary cross-entropy, edge loss (boundary-weighted BCE), and an uncertainty regularization term.
- **Stochastic Weight Averaging (SWA)**: Averages model weights across the final training epochs for flatter minima and improved generalization.
- **Automatic Mixed Precision (AMP)**: Reduces memory footprint and accelerates training on GPU.
- **Learning Rate Scheduling**: Cosine Annealing with warm restarts.
- **Optimizer**: AdamW with weight decay for Transformer-friendly optimization.
- **Gradient Accumulation**: Effective batch scaling (4 accumulation steps) for memory-constrained hardware.

### 5) Model Evaluation

Models are evaluated using standard camouflaged object detection metrics:

- **S-measure (Sm)**: Structural similarity between predicted and ground-truth masks.
- **E-measure (Em)**: Enhanced alignment measure capturing both global and local fidelity.
- **Weighted F-measure (wFm)**: Boundary-aware F-measure that penalizes coarse predictions.
- **IoU**: Intersection-over-Union for region-level accuracy.
- **F1-Score**: Harmonic mean of precision and recall at the pixel level.
- **MAE**: Mean Absolute Error for per-pixel prediction calibration.

### 6) Ablation Studies

To analyze the contribution of individual design choices, controlled ablation studies are conducted examining ensemble composition and architectural diversity:

- **Individual vs. Same-Family Ensemble vs. Cross-Family Ensemble** – to quantify the benefit of heterogeneous architectural pairing (Transformers + CNNs) over homogeneous ensembles.
- **Confidence-Weighted vs. Mean vs. Max Fusion** – to evaluate the impact of adaptive ensemble aggregation strategies.
- **Frequency Branch** – to assess whether FFT-domain saliency provides complementary signal beyond spatial features.
- **Edge Auxiliary Head** – to measure the contribution of boundary-supervised training to final mask sharpness.
- **Uncertainty Head** – to study the role of MC Dropout regularization on prediction calibration.
- **SWA vs. Standard Training** – to evaluate weight-averaging stability improvements across all backbones.

These studies clarify that cross-family architectural diversity is the dominant driver of ensemble gain, and that the frequency and edge auxiliary heads each contribute meaningful improvements to boundary-level metrics.

### 7) Explainable AI (XAI) Framework

Model transparency is ensured through multiple complementary interpretability techniques:

- **Grad-CAM**: Highlights spatial attention regions from CNN and Transformer backbones, confirming predictions are anchored to the camouflaged object rather than background distractors.
- **Confidence Attribution Maps**: Pixel-wise visualization of which backbone (Swin-Base vs. ResNet50) drives the ensemble prediction in each image region, revealing complementary specialization.
- **Frequency Saliency (FFT)**: Compares spectral saliency between Swin-Base, ResNet50, and the ensemble, showing how different bands contribute complementary cues.
- **Error Reduction Analysis**: Heatmap comparison of per-pixel errors across individual models and the ensemble, demonstrating where ensemble fusion corrects failures that both individual models share.
- **Integrated Gradients (IG)**: Path-integrated input attribution pinpoints exact pixel contributions to the segmentation score, providing axiomatic feature attribution across the full decoder.
- **LIME**: Superpixel-level local explanations indicate which image regions positively or negatively support the detection decision, independently of model internals.
- **Failure Mode UMAP**: Dimensionality-reduced embedding of hard cases (IoU < 0.3) visualizes how ensemble fusion redistributes and reduces the hard-case concentration relative to individual models.
- **MC Dropout Uncertainty Maps**: Per-pixel variance across stochastic forward passes identifies ambiguous boundary regions and total false-detection zones for human-in-the-loop review.

## Results

The system demonstrates strong and consistent camouflaged object detection performance across the heterogeneous 4-model ensemble:

**Key Findings:**

- The **4-Model Heterogeneous Ensemble** achieves the best S-measure (0.8612 confidence-weighted), outperforming all individual backbones and same-family sub-ensembles.
- **Cross-family diversity** (Transformer + CNN) consistently outperforms same-family ensembling — the Transformer ensemble (Sm 0.8551) and CNN ensemble (Sm 0.8413) each individually fall below the best individual model (Swin-Base, Sm 0.8609), while the full 4-model ensemble surpasses all.
- **Swin-Base** achieves the highest individual S-measure (0.8609) and SWA score (0.8890), demonstrating strong global context modeling for camouflage.
- **PoolFormer-S36** achieves SWA Sm 0.8444; **ResNet50** SWA Sm 0.8407; **EfficientNet-B4** SWA Sm 0.8199.
- The **frequency branch** provides complementary spectral cues that differ systematically between Transformer and CNN backbones, with ensemble FFT saliency being richer than any single model.
- **Ensemble error reduction** is visible across all tested images: boundary hallucinations and interior miss-detections present in both individual models are substantially corrected by confidence-weighted fusion.
- **Few-shot adaptation** (10% → 50% target training data) shows consistent metric gains across all indicators: Em rises from 0.827 to 0.876, IoU from 0.581 to 0.624, confirming strong transfer from pretrained representations.
- **XAI visualizations** confirm predictions are driven by object texture and boundary characteristics rather than irrelevant background regions.

## Dataset

**Sources:**

- [COD10K](https://github.com/DengPingFan/SINet) — 10,000 images across 78 camouflaged object categories.
- [CAMO](https://sites.google.com/view/camo-dataset) — 1,250 camouflaged and 1,250 non-camouflaged images across 8 categories.

**Task**: Binary segmentation (camouflaged object vs. background)

**Type**: RGB images with pixel-wise ground-truth masks

## Dependencies

The project uses the following Python libraries:

```
numpy
torch
torchvision
timm
albumentations
opencv-python
scipy
matplotlib
captum
lime
scikit-learn
```

Install dependencies with:

```bash
pip install torch torchvision timm albumentations opencv-python scipy matplotlib captum lime scikit-learn
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/sakshammgarg/CamouflageDetection.git
cd CamouflageDetection
```

2. Download the datasets:
   - Visit [COD10K](https://www.kaggle.com/datasets/getcam/cod10k) and [CAMO](https://www.kaggle.com/datasets/ivanomelchenkoim11/camo-dataset).
   - Download and extract both datasets.
   - Organize images as follows:

```
/input
  /cod10k
    /Train
      /Images
      /GT_Object
    /Test
      /Images
      /GT_Object
  /camo
    /Train
    /Test
```

## Usage

Run the provided notebook to execute the full training and evaluation pipeline.

The notebook performs the following steps:

1. Initializes the novel CamouflageAug, TextureMix, FrequencyJitter, and CutMixSeg augmentation pipeline.
2. Trains four backbone models (Swin-Base, PoolFormer-S36, ResNet50, EfficientNet-B4) with composite loss and SWA.
3. Constructs the confidence-weighted heterogeneous ensemble.
4. Displays evaluation metrics (Sm, Em, wFm, IoU, F1, MAE).
5. Generates Grad-CAM, confidence attribution, frequency saliency, Integrated Gradients, and LIME visualizations.
6. Produces MC Dropout uncertainty maps and failure mode UMAP analyses.
7. Evaluates cross-dataset generalization and few-shot adaptation curves.

## Model Selection Guide

**Recommended Choices for Deployment:**

- **Highest Accuracy**: 4-Model Heterogeneous Ensemble (confidence-weighted).
- **Single-Model Best**: Swin-Base (SWA Sm 0.8890).
- **Edge or Low-Power Devices**: EfficientNet-B4 or PoolFormer-S36.
- **Boundary-Critical Applications**: Ensemble with Edge Auxiliary Head enabled.
- **Uncertainty-Aware Deployment**: Any backbone with MC Dropout active at inference.
- **Clear Visual Explanations**: ResNet50 with Grad-CAM.

## Advanced Features

Key technical highlights include:

- **Heterogeneous Ensemble Design**: Cross-family pairing of Transformer and CNN backbones to maximize complementary feature coverage.
- **Novel CamouflageAug**: Domain-specific augmentation that simulates foreground-background blending at the pixel level.
- **Frequency-Domain Branch**: FFT-based saliency head providing spectral cues complementary to spatial features.
- **Edge-Guided Auxiliary Supervision**: Boundary-weighted loss head for sharper contour predictions.
- **MC Dropout Uncertainty Estimation**: Calibrated pixel-wise uncertainty maps via stochastic inference.
- **Confidence-Weighted Ensemble Fusion**: Adaptive aggregation driven by per-pixel model certainty.
- **Stochastic Weight Averaging (SWA)**: Improves generalization by averaging weights over training trajectory.
- **Composite Loss Function**: Multi-term objective combining IoU, Dice, BCE, edge, and uncertainty losses.
- **Multi-Method XAI Suite**: Grad-CAM, Integrated Gradients, LIME, frequency saliency, error attribution, and UMAP failure analysis.
- **Cross-Dataset Few-Shot Evaluation**: Systematic assessment of transfer efficiency from 10% to 50% target training data.

## Applications

Potential real-world applications include:

- Wildlife Conservation and Anti-Poaching Surveillance
- Military Target Detection and Reconnaissance
- Medical Image Segmentation (polyp and lesion detection)
- Industrial Inspection for Surface Defect Detection
- Underwater Robotics and Marine Organism Monitoring
- Search and Rescue Operations in Cluttered Environments

## Future Improvements

Planned extensions of this work include:

1. **Video Camouflage Detection**: Leveraging temporal consistency across frames for moving camouflaged targets.
2. **Severity-Aware Ranking**: Scoring detections by camouflage difficulty for prioritized review.
3. **Knowledge Distillation**: Compressing the 4-model ensemble into a single deployable student network.
4. **3D and Multi-Modal Fusion**: Incorporating depth or thermal imagery for challenging low-visibility scenarios.
5. **Foundation Model Integration**: Fine-tuning SAM (Segment Anything Model) as an additional ensemble member.

---

This project demonstrates how heterogeneous deep learning ensembles, frequency-domain feature analysis, and a multi-method explainable AI suite can be combined to build accurate, robust, and trustworthy systems for automated camouflaged object detection.
