# Skill: Shadow Removal in Computer Vision

## Metadata
- **Knowledge Domain**: Shadow Detection & Removal, Image Restoration, Computational Photography
- **Number of Sources**: 26 documents
- **Last Updated**: 2026-04-13
- **Target Agent Type**: Research assistant, shadow removal method advisor, CV paper review bot

---

## Overview

The knowledge base primarily covers **single-image shadow removal** [Source: Single-Image Shadow Removal Using Deep Learning_ A Comprehensive Survey.pdf, chunk 16]. Key sub-problems addressed include **shadow detection, localization, and the subsequent removal of shadows** [Source: SpA-Former_An Effective and lightweight Transformer for image shadow removal.pdf, chunk 5; Source: WSRD_A Novel Benchmark for High Resolution Image Shadow Removal.pdf, chunk 16]. Challenges involve preserving initial image details and accounting for spatial correlation in remote areas [Source: SpA-Former_An Effective and lightweight Transformer for image shadow removal.pdf, chunk 6]. The context does not provide information on video or document shadow removal.

Methodologies discussed range from **traditional image processing techniques** (e.g., gradient, illumination priors) [Source: SpA-Former_An Effective and lightweight Transformer for image shadow removal.pdf, chunk 5; Source: Boundary-Aware Divide and Conquer A Diffusion-based Solution for Unsupervised Shadow Removal.pdf, chunk 17], which often produce boundary artifacts, to more recent **deep learning-based approaches**. These include **Convolutional Neural Networks (CNNs)** [Source: WSRD_A Novel Benchmark for High Resolution Image Shadow Removal.pdf, chunk 16], **GAN-based self-supervised models, mask-assisted semi-supervised models, and unsupervised learning methods** [Source: SpA-Former_An Effective and lightweight Transformer for image shadow removal.pdf, chunk 5]. **Transformer-based architectures** such as SpA-Former and Shadowformer are also explored [Source: SpA-Former_An Effective and lightweight Transformer for image shadow removal.pdf, chunk 6; Source: ShadowRefiner_Towards Mask-free Shadow Removal via Fast Fourier.pdf, chunk 69]. The overall research landscape indicates a shift from traditional methods to deep learning, often employing a two-stage process of detection followed by removal, leveraging large datasets to boost performance [Source: SpA-Former_An Effective and lightweight Transformer for image shadow removal.pdf, chunk 6; Source: Boundary-Aware Divide and Conquer A Diffusion-based Solution for Unsupervised Shadow Removal.pdf, chunk 17].

---

## Core Concepts

Here are several important technical concepts in shadow removal research, based on the provided context:

1.  **Shadow Removal as Image Restoration**: Shadow removal is categorized as a specific case of image restoration, addressing degradations like altered illumination, color, detail, and noise levels caused by shadows [Source: WSRD_A Novel Benchmark for High Resolution Image Shadow Removal.pdf, chunk 7].
2.  **Shadow Detection**: This is often the initial stage in many de-shadowing methods, where the regions covered by shadows are identified [Source: SpA-Former_An Effective and lightweight Transformer for image shadow removal.pdf, chunk 6; Single-Image Shadow Removal Using Deep Learning_ A Comprehensive Survey.pdf, chunk 188].
3.  **Two-stage Shadow Removal Methods**: Many current image de-shadowing approaches are implemented in two stages, typically involving an initial step of detecting shadows [Source: SpA-Former_An Effective and lightweight Transformer for image shadow removal.pdf, chunk 6].
4.  **Preservation of Image Details**: A key challenge in shadow removal is to effectively eliminate shadows without compromising or distorting the original fine details present in the image [Source: SpA-Former_An Effective and lightweight Transformer for image shadow removal.pdf, chunk 6].
5.  **Physics-based Models**: This refers to methods for shadow removal that incorporate physical models, which likely account for the real-world formation of shadows due to light-object interactions [Source: Single-Image Shadow Removal Using Deep Learning_ A Comprehensive Survey.pdf, chunk 188; DenseSR_Image Shadow Removal as Dense Prediction.pdf, chunk 9].
6.  **User-defined Selective Shadow Removal**: Users often seek the flexibility to remove only specific parts of shadows or adjust their intensity in particular areas, guided by their input cues [Source: Single-Image Shadow Removal Using Deep Learning_ A Comprehensive Survey.pdf, chunk 177].
7.  **Shadow-induced Degradations**: Shadows can introduce a range of degradations to image properties, including changes in illumination, color, detail, and noise levels, altering 3D scene observations [Source: WSRD_A Novel Benchmark for High Resolution Image Shadow Removal.pdf, chunk 7].
8.  **Model Architecture Design**: This encompasses the structural development of the computational models or networks used to tackle the complexities of shadow removal [Source: Single-Image Shadow Removal Using Deep Learning_ A Comprehensive Survey.pdf, chunk 188].
9.  **Training Strategies**: These are the various methodologies and approaches employed to train machine learning models for effective shadow removal [Source: Single-Image Shadow Removal Using Deep Learning_ A Comprehensive Survey.pdf, chunk 188].
10. **Spatial Correlation**: This refers to the relationship between remote areas within an image, which current network architectures often struggle to account for during shadow removal [Source: SpA-Former_An Effective and lightweight Transformer for image shadow removal.pdf, chunk 6].
11. **Shadow Mask Acquisition**: The process of generating a shadow mask, which can be done by calculating the difference between a shadowed image and its corresponding shadow-free image from a training dataset [Source: NTIRE 2024 Image Shadow Removal Challenge Report.pdf, chunk 117].
12. **Impact on Downstream Vision Tasks**: The presence of shadows negatively affects the performance of various other computer vision applications, such as object recognition, tracking, image segmentation, and 3D reconstruction [Source: WSRD_A Novel Benchmark for High Resolution Image Shadow Removal.pdf, chunk 7; DenseSR_Image Shadow Removal as Dense Prediction.pdf, chunk 9].

---

## Key Trends

The most significant research trends and recent breakthroughs in shadow removal are predominantly driven by deep learning advancements.

Here are 7 trends:

1.  **Shift from Traditional to Deep Learning Approaches:** While classic methods relied on prior information like gradients, illumination, and region consistency, often leading to boundary artifacts in real-world scenarios, recent deep learning-based methods have significantly boosted removal performance, often leveraging large-scale datasets. Traditional methods such as Guo et al. (2012) have been largely superseded by deep learning solutions like DeshadowNet and DHAN [Source: Boundary-Aware Divide and Conquer A Diffusion-based Solution for Unsupervised Shadow Removal.pdf, chunk 17; SpA-Former_An Effective and lightweight Transformer for image shadow removal.pdf, chunk 5; Diff-Shadow_Global-guided Diffusion Model for Shadow Removal.pdf, chunk 58].
2.  **Emergence of Unsupervised and Self-supervised Methods:** There is a growing trend towards unsupervised learning methods and self-supervised models (often based on GANs) to address the challenges of acquiring large, labeled datasets for shadow removal. Semi-supervised models utilizing mask-assisted guidance are also noted [Source: SpA-Former_An Effective and lightweight Transformer for image shadow removal.pdf, chunk 5].
3.  **GAN-based Methods:** Generative Adversarial Networks (GANs) are a prominent deep learning approach, particularly in self-supervised models for shadow removal. ST-CGAN is cited as an example of a deep learning-based method in this category [Source: SpA-Former_An Effective and lightweight Transformer for image shadow removal.pdf, chunk 5; Diff-Shadow_Global-guided Diffusion Model for Shadow Removal.pdf, chunk 58].
4.  **Transformer-based Architectures:** Transformer models are being increasingly adopted, with methods like ShadowFormer representing a recent application of this architecture for image shadow removal [Source: Diff-Shadow_Global-guided Diffusion Model for Shadow Removal.pdf, chunk 58].
5.  **Introduction of Diffusion Models:** Diffusion models represent a very recent and significant breakthrough. As of late 2022, this area was nascent, with approaches like ShadowDiffusion beginning to exploit these models for shadow removal tasks [Source: ShadowDiffusion_When Degradation Prior Meets Diffusion Model for Shadow Removal.pdf, chunk 6; Diff-Shadow_Global-guided Diffusion Model for Shadow Removal.pdf, chunk 58].
6.  **Addressing Limitations in Shadow Degradation Prior:** A recognized limitation in existing deep shadow removal methods is the inadequate exploitation of the shadow degradation prior, which reflects the physical properties of shadows. Current models often assume overly restrictive shadow models (e.g., linear and uniform degradation) that are insufficient for real-world complicated lighting conditions [Source: ShadowDiffusion_When Degradation Prior Meets Diffusion Model for Shadow Removal.pdf, chunk 9].
7.  **Emphasis on Shadow Detection and Localization:** An underlying prerequisite for effective shadow elimination is the accurate detection and localization of shadows within an image [Source: SpA-Former_An Effective and lightweight Transformer for image shadow removal.pdf, chunk 5].

The provided context does not contain enough information regarding specific trends or breakthroughs in video shadow removal or document shadow removal.

---

## Key Entities

Based on the provided context:

**Authors/Groups:**
*   Wang, Li, and Yang (2018), who proposed the ISTD dataset [Source: Efficient Model-Driven Network for Shadow Removal.pdf, chunk 46].
*   Qu et al. (2017), who proposed the SRD dataset [Source: Efficient Model-Driven Network for Shadow Removal.pdf, chunk 46].
*   Zhu et al. (2022b) [Source: OmniSR_Shadow Removal Under Direct and Indirect Lighting.pdf, chunk 57].
*   Vasluianu, Seizinger, and Timofte (2023), associated with the WRSD+ dataset [Source: OmniSR_Shadow Removal Under Direct and Indirect Lighting.pdf, chunk 57].
*   Vasluianu et al. (2024), associated with the NTIRE 2024 Image Shadow Removal Challenge [Source: OmniSR_Shadow Removal Under Direct and Indirect Lighting.pdf, chunk 57].
*   Hu et al. (2019a) [Source: OmniSR_Shadow Removal Under Direct and Indirect Lighting.pdf, chunk 57].
*   Cun, Pun, and Shi (2020) [Source: OmniSR_Shadow Removal Under Direct and Indirect Lighting.pdf, chunk 57].
*   Fu et al. (2021c) [Source: OmniSR_Shadow Removal Under Direct and Indirect Lighting.pdf, chunk 57].

**Datasets/Benchmarks:**
*   **ISTD:** A real-world shadow-removal benchmark, and the first public benchmark for training shadow detection and removal. It consists of 1870 image triplets (shadow image, shadow mask, and shadow-free image) across 135 various scenes. It is divided into 1330 triplets for training and 540 triplets for testing. The shadow mask is derived from the binary difference between the shadow and shadow-free images [Source: ShadowDiffusion_When Degradation Prior Meets Diffusion Model for Shadow Removal.pdf, chunk 51], [Source: Latent Feature-Guided Diffusion Models for Shadow Removal.pdf, chunk 48], [Source: Efficient Model-Driven Network for Shadow Removal.pdf, chunk 46].
*   **Adjusted ISTD (ISTD+ / AISTD):** This dataset reduces the illumination inconsistency between the shadow and shadow-free images present in the original ISTD dataset [Source: ShadowDiffusion_When Degradation Prior Meets Diffusion Model for Shadow Removal.pdf, chunk 51], [Source: Latent Feature-Guided Diffusion Models for Shadow Removal.pdf, chunk 48].
*   **SRD:** Consists of 2680 training pairs and 408 testing pairs of shadow and shadow-free images [Source: ShadowDiffusion_When Degradation Prior Meets Diffusion Model for Shadow Removal.pdf, chunk 51], [Source: Efficient Model-Driven Network for Shadow Removal.pdf, chunk 46].
*   **WRSD+:** This dataset does not provide testing data directly. Its evaluation data and code are used from the NTIRE 2024 Image Shadow Removal Challenge for comparison [Source: OmniSR_Shadow Removal Under Direct and Indirect Lighting.pdf, chunk 57].
*   **NTIRE 2024 Image Shadow Removal Challenge:** Provides evaluation data and code, particularly for datasets like WRSD+ [Source: OmniSR_Shadow Removal Under Direct and Indirect Lighting.pdf, chunk 57].

The context also highlights a clear need for a comprehensive benchmark that includes large-scale test samples, diverse indoor and outdoor scenes, and challenging lighting conditions (e.g., various light sources, multiple shadows) for fair and rigorous evaluation of shadow removal methods [Source: Single-Image Shadow Removal Using Deep Learning_ A Comprehensive Survey.pdf, chunk 179], [Source: Single-Image Shadow Removal Using Deep Learning_ A Comprehensive Survey.pdf, chunk 178].

**Key Methods/Baselines:**
*   DSC (Hu et al. 2019a) [Source: OmniSR_Shadow Removal Under Direct and Indirect Lighting.pdf, chunk 57]
*   DHAN (Cun, Pun, and Shi 2020) [Source: OmniSR_Shadow Removal Under Direct and Indirect Lighting.pdf, chunk 57]
*   Fu et al. (Fu et al. 2021c) [Source: OmniSR_Shadow Removal Under Direct and Indirect Lighting.pdf, chunk 57]
*   Zhu et al. (Zhu et al. 2022b) [Source: OmniSR_Shadow Removal Under Direct and Indirect Lighting.pdf, chunk 57]

The context does not contain enough information regarding LRSS or Video Shadow datasets.

---

## Methodology & Best Practices

The dominant methodologies in shadow removal encompass both classic and deep learning-based approaches, employing various pipelines and training strategies.

**(1) Detection-then-removal pipelines vs. end-to-end methods:**
Current methods for image de-shadowing are almost always two-stage, involving detecting the shadow first [Source: SpA-Former_An Effective and lightweight Transformer for image shadow removal.pdf, chunk 2]. The context implies that preserving initial image details while removing shadows is challenging, and present networks may not fully account for spatial correlation between remote areas [Source: SpA-Former_An Effective and lightweight Transformer for image shadow removal.pdf, chunk 2].

**(2) Physics-based vs. data-driven approaches:**
Classic shadow removal methods rely on prior information such as gradient, illumination, and region consistency. These methods are built on assumptions of ideal conditions, which can lead to noticeable shadow boundary artifacts in real-world scenarios [Source: Boundary-Aware Divide and Conquer A Diffusion-based Solution for Unsupervised Shadow Removal.pdf, chunk 17]. In contrast, recent deep learning-based shadow removal methods improve performance by utilizing large-scale datasets [Source: Boundary-Aware Divide and Conquer A Diffusion-based Solution for Unsupervised Shadow Removal.pdf, chunk 17]. Examples of deep learning-based methods include De-shadowNet, ST-CGAN, DHAN, DC-ShadowNet, BM-Net, ShadowFormer, and ShadowDiffusion, alongside traditional methods like Guo et al. (2012) [Source: Diff-Shadow_Global-guided Diffusion Model for Shadow Removal.pdf, chunk 58]. Deep learning methods, particularly neural networks, offer powerful representation ability [Source: SpA-Former_An Effective and lightweight Transformer for image shadow removal.pdf, chunk 3].

**(3) How attention, multi-scale features, and context are used:**
The provided context does not contain sufficient information to describe how attention, multi-scale features, and context are specifically used as methodologies in shadow removal. While one of the major issues mentioned is the incapability of current networks to account for spatial correlation between remote areas [Source: SpA-Former_An Effective and lightweight Transformer for image shadow removal.pdf, chunk 2], the context does not detail the specific techniques employed to address this.

**(4) Common training strategies and loss functions used in shadow removal:**
Deep learning-based shadow removal methods utilize various training strategies, including self-supervised models based on Generative Adversarial Networks (GANs), semi-supervised models that incorporate mask-assisted guidance, and unsupervised learning methods [Source: SpA-Former_An Effective and lightweight Transformer for image shadow removal.pdf, chunk 3]. Loss functions are a component discussed in surveys of single-image shadow removal methods [Source: Single-Image Shadow Removal Using Deep Learning_ A Comprehensive Survey.pdf, chunk 6], but the provided context does not describe common or specific loss functions used.

---

## Diffusion-Based Methods

Based on the provided context, the following diffusion-based shadow removal methods are mentioned:

1.  **Diff-Shadow:**
    *   **(1) Key technical innovation:** Diff-Shadow is a "Global-guided Diffusion Model for Shadow Removal" designed to improve the quality of shadow removal during the diffusion denoising process [Source: Diff-Shadow_Global-guided Diffusion Model for Shadow Removal.pdf, chunk 58, chunk 17].
    *   **(2) How it uses diffusion models differently from GANs:** The context does not explicitly detail how Diff-Shadow's use of diffusion models differs from GANs.
    *   **(3) Datasets it was evaluated on:** ISTD (Wang, Li, and Yang 2018), ISTD+ (Le and Samaras 2019), and SRD (Qu et al. 2017) datasets [Source: Diff-Shadow_Global-guided Diffusion Model for Shadow Removal.pdf, chunk 17].
    *   **(4) Main advantage over prior methods:** Diff-Shadow "outperforms the state of the art" [Source: Diff-Shadow_Global-guided Diffusion Model for Shadow Removal.pdf, chunk 17].

2.  **ShadowDiffusion:**
    *   **(1) Key technical innovation:** ShadowDiffusion is an "unrolling diffusion-based shadow removal framework" that integrates both generative and degradation priors. It formulates the shadow removal problem as jointly pursuing the shadow-free image and a refined shadow mask, where mask refinement is an auxiliary task of the diffusion generator that progressively refines the mask along with shadow-free image restoration in an interactive manner [Source: ShadowDiffusion_When Degradation Prior Meets Diffusion Model for Shadow Removal.pdf, chunk 12]. It is noted as pioneering the integration of diffusion models into the domain of shadow removal [Source: Single-Image Shadow Removal Using Deep Learning_ A Comprehensive Survey.pdf, chunk 103].
    *   **(2) How it uses diffusion models differently from GANs:** The context does not explicitly detail how ShadowDiffusion's use of diffusion models differs from GANs.
    *   **(3) Datasets it was evaluated on:** The context does not mention the specific datasets ShadowDiffusion was evaluated on.
    *   **(4) Main advantage over prior methods:** ShadowDiffusion pioneered the integration of diffusion models into shadow removal [Source: Single-Image Shadow Removal Using Deep Learning_ A Comprehensive Survey.pdf, chunk 103].

3.  **Latent Feature-Guided Diffusion Models for Shadow Removal:**
    *   **(1) Key technical innovation:** This method proposes the use of diffusion models as a promising approach to gradually refine the details of images for shadow removal [Source: Latent Feature-Guided Diffusion Models for Shadow Removal.pdf, chunk 5]. The specific "latent feature-guided" aspect is mentioned in the title but not further elaborated in the provided context.
    *   **(2) How it uses diffusion models differently from GANs:** It generally states that "diffusion models beat gans on image sy" (synthesis) [Source: Latent Feature-Guided Diffusion Models for Shadow Removal.pdf, chunk 4], and that diffusion models offer a "promising approach to gradually refine the details" [Source: Latent Feature-Guided Diffusion Models for Shadow Removal.pdf, chunk 5].
    *   **(3) Datasets it was evaluated on:** The context does not mention the specific datasets this method was evaluated on.
    *   **(4) Main advantage over prior methods:** It leverages diffusion models as a promising approach for detail refinement in shadow removal [Source: Latent Feature-Guided Diffusion Models for Shadow Removal.pdf, chunk 5].

The methods "Detail-Preserving Latent Diffusion" and "Boundary-Aware Divide and Conquer" mentioned in the question are not described in the provided context passages. While the context generally mentions that "a series of methods [27, 39, 62] tend to employ the diffusion model as the backbone" [Source: Single-Image Shadow Removal Using Deep Learning_ A Comprehensive Survey.pdf, chunk 103], it does not elaborate on their specific technical innovations, comparison to GANs, datasets, or main advantages.

---

## Benchmark Datasets Comparison

Here's a comparison of the main shadow removal benchmark datasets based on the provided context:

**1. ISTD (Image Shadow Detection and Removal Dataset)**
*   **Number of Samples:** Consists of 1870 image triplets, divided into 1330 for training and 540 for testing [Source: ShadowDiffusion_When Degradation Prior Meets Diffusion Model for Shadow Removal.pdf, chunk 51; Efficient Model-Driven Network for Shadow Removal.pdf, chunk 46; Latent Feature-Guided Diffusion Models for Shadow Removal.pdf, chunk 48; Shadow Removal by a Lightness-Guided Network With Training on Unpaired Data.pdf, chunk 70].
*   **Shadow Masks:** Yes, it provides triplets consisting of a shadow image, a corresponding shadow mask, and a shadow-free image [Source: ShadowDiffusion_When Degradation Prior Meets Diffusion Model for Shadow Removal.pdf, chunk 51; Efficient Model-Driven Network for Shadow Removal.pdf, chunk 46; Latent Feature-Guided Diffusion Models for Shadow Removal.pdf, chunk 48; Shadow Removal by a Lightness-Guided Network With Training on Unpaired Data.pdf, chunk 70]. The shadow mask is extracted from the binary difference between the shadow image and the shadow-free image [Source: Latent Feature-Guided Diffusion Models for Shadow Removal.pdf, chunk 48].
*   **Scene Types:** It is a real-world shadow removal benchmark with 135 various scenes [Source: Latent Feature-Guided Diffusion Models for Shadow Removal.pdf, chunk 48; Efficient Model-Driven Network for Shadow Removal.pdf, chunk 46]. It shows good variety in terms of illumination, shape, and scene [Source: Shadow Removal by a Lightness-Guided Network With Training on Unpaired Data.pdf, chunk 70]. Specific indoor/outdoor details are not provided.
*   **Resolution:** Not specified in the context.
*   **Known Limitations:** Not explicitly detailed in the context, other than the illumination inconsistency that ISTD+ aims to address.

**2. Adjusted ISTD (ISTD+ / AISTD)**
*   **Number of Samples:** The context implies it is based on the ISTD dataset [Source: ShadowDiffusion_When Degradation Prior Meets Diffusion Model for Shadow Removal.pdf, chunk 51; Shadow Removal by a Lightness-Guided Network With Training on Unpaired Data.pdf, chunk 70]. Specific training/test splits are not separately provided, suggesting it uses the same base structure as ISTD.
*   **Shadow Masks:** Not explicitly stated, but as an adjusted version of ISTD, it likely retains or derives masks from the original ISTD data.
*   **Scene Types:** Inherits characteristics from ISTD.
*   **Resolution:** Not specified in the context.
*   **Known Limitations:** It addresses and reduces the illumination inconsistency between the shadow and shadow-free image found in the original ISTD dataset [Source: ShadowDiffusion_When Degradation Prior Meets Diffusion Model for Shadow Removal.pdf, chunk 51; Shadow Removal by a Lightness-Guided Network With Training on Unpaired Data.pdf, chunk 70].

**3. SRD (Shadow Removal Dataset)**
*   **Number of Samples:** Consists of 2680 pairs for training and 408 pairs for testing [Source: ShadowDiffusion_When Degradation Prior Meets Diffusion Model for Shadow Removal.pdf, chunk 51; Efficient Model-Driven Network for Shadow Removal.pdf, chunk 46].
*   **Shadow Masks:** The dataset provides pairs of shadow and shadow-free images [Source: ShadowDiffusion_When Degradation Prior Meets Diffusion Model for Shadow Removal.pdf, chunk 51; Efficient Model-Driven Network for Shadow Removal.pdf, chunk 46]. While the context mentions using "predicted masks" in some experiments for SRD [Source: ShadowDiffusion_When Degradation Prior Meets Diffusion Model for Shadow Removal.pdf, chunk 51] and also mentions "ground-truth masks" in relation to SRD [Source: OmniSR_Shadow Removal Under Direct and Indirect Lighting.pdf, chunk 65], the direct provision of ground-truth masks with *every* image pair as part of the original dataset structure (like ISTD's triplets) is not explicitly detailed.
*   **Scene Types:** Not specified in the context.
*   **Resolution:** Not specified in the context.
*   **Known Limitations:** Not specified in the context.

**4. WSRD+ Dataset**
*   **Number of Samples:** The context mentions "evaluation data" for WSRD+ [Source: Detail-Preserving Latent Diffusion for Stable Shadow Removal.pdf, chunk 60], but specific training/testing sample counts are not provided.
*   **Shadow Masks:** Not specified in the context.
*   **Scene Types:** Not specified in the context.
*   **Resolution:** Not specified in the context.
*   **Known Limitations:** Not specified in the context.

**5. Other Datasets**
*   **INS Dataset:** Mentioned as a dataset where a method achieved high scores, alongside ISTD, ISTD+, and SRD [Source: OmniSR_Shadow Removal Under Direct and Indirect Lighting.pdf, chunk 65]. No further details are provided.

**6. NTIRE Challenge Datasets**
*   The context mentions the **NTIRE 2024 Image Shadow Removal Challenge** [Source: Detail-Preserving Latent Diffusion for Stable Shadow Removal.pdf, chunk 60]. This challenge provided evaluation data and evaluation code for the WSRD+ dataset [Source: Detail-Preserving Latent Diffusion for Stable Shadow Removal.pdf, chunk 60].
*   There is no mention of NTIRE 2023 or 2025 challenge datasets in the provided context.

---

## Mask-Free Shadow Removal Methods

Based on the provided context, the following shadow removal methods do not require a shadow mask as input:

1.  **Polarization-guided (referred to as "Ours")**
    *   **Mask-free status:** This method explicitly states it does not require an externally obtained shadow mask, unlike many other methods such as Inpaint4Shadow [Source: Polarization Guided Mask-Free Shadow Removal.pdf, chunk 13].
    *   **(1) How it handles shadows without a mask:** Instead of a shadow mask, it uses "Polarization guidance," specifically leveraging "DoP" (Degree of Polarization) and other "Extra information" directly from the shadow image [Source: Polarization Guided Mask-Free Shadow Removal.pdf, chunk 13].
    *   **(2) Its architecture:** The provided context refers to it as "Polarization guidance Ours" but does not detail its specific architectural components [Source: Polarization Guided Mask-Free Shadow Removal.pdf, chunk 13].
    *   **(3) Performance comparison:** The context highlights that mask-requiring methods often introduce additional workload and can lead to degenerated performance near shadow boundaries due to mask inaccuracy. While it implies the "Polarization guidance Ours" method avoids these issues, it does not provide a direct comparative performance metric against mask-guided methods [Source: Polarization Guided Mask-Free Shadow Removal.pdf, chunk 13].

2.  **"Our method" from OmniSR (unnamed in provided context)**
    *   **Mask-free status:** This method (from the OmniSR paper) does not use ground-truth shadow masks as input, distinguishing it from methods like DMTN, ShadowFormer, ShadowDiffusion, BMNet, and Fu et al., which typically rely on them [Source: OmniSR_Shadow Removal Under Direct and Indirect Lighting.pdf, chunk 64].
    *   **(1) How it handles shadows without a mask:** The context states that "Our method" offers "more comprehensive shadow removal, even in complex scenes," including under direct and indirect lighting, but does not elaborate on the specific mechanism for handling shadows without a mask [Source: OmniSR_Shadow Removal Under Direct and Indirect Lighting.pdf, chunk 64].
    *   **(2) Its architecture:** The context does not provide any details about the architecture of this method [Source: OmniSR_Shadow Removal Under Direct and Indirect Lighting.pdf, chunk 64].
    *   **(3) Performance comparison:** It "demonstrates more comprehensive shadow removal, even in complex scenes" compared to methods like DMTN, ShadowFormer, ShadowDiffusion, BMNet, and Fu et al., which do use ground-truth shadow masks [Source: OmniSR_Shadow Removal Under Direct and Indirect Lighting.pdf, chunk 64]. This suggests a superior or more robust performance despite not relying on a mask.

The context mentions HomoFormer in visual comparisons [Source: Single-Image Shadow Removal Using Deep Learning_ A Comprehensive Survey.pdf, chunks 150, 145] but does not specify whether it requires a shadow mask as input, nor does it detail its architecture or how it handles shadows without one. Methods like ShadowRefiner and PhaSR are not mentioned in the provided context passages. The context explicitly states that Inpaint4Shadow requires an externally obtained shadow mask [Source: Polarization Guided Mask-Free Shadow Removal.pdf, chunk 13].

---

## Knowledge Gaps & Limitations

Open challenges and limitations in shadow removal research include:

*   **Over-smoothing artifacts**: A major challenge is removing shadows while simultaneously preserving the initial image details, as much information is hidden by shadows [Source: SpA-Former_An Effective and lightweight Transformer for image shadow removal.pdf, chunk 6].
*   **Color inconsistency**: Restoring the authentic appearance of occluded regions is a fundamental computer vision task, implying that achieving color consistency is a key goal and a potential challenge [Source: DenseSR_Image Shadow Removal as Dense Prediction.pdf, chunk 9].
*   **Generalization to real-world shadows**: Current deep shadow removal models struggle to achieve satisfactory performance when faced with shadow images captured in real-world scenarios [Source: Single-Image Shadow Removal Using Deep Learning_ A Comprehensive Survey.pdf, chunk 6]. This is further complicated by the need to handle diverse scenes (indoor/outdoor) and challenging lighting conditions, including various light sources and multiple shadows [Source: Single-Image Shadow Removal Using Deep Learning_ A Comprehensive Survey.pdf, chunk 4]. Existing datasets may not fully capture the complexity of real shadow degradations [Source: Single-Image Shadow Removal Using Deep Learning_ A Comprehensive Survey.pdf, chunk 6].
*   **Video temporal consistency**: The provided context does not contain information regarding video temporal consistency in shadow removal.
*   **Evaluation metric limitations**: There is a recognized need for "Non-Reference Evaluation Metrics" and for rigorous evaluations of shadow removal methods, suggesting current evaluation methods have limitations, particularly concerning the difficulty of collecting paired shadow and shadow-free images [Source: Single-Image Shadow Removal Using Deep Learning_ A Comprehensive Survey.pdf, chunk 4].
*   **Lack of large-scale real paired datasets**: A significant limitation is the lack of a comprehensive benchmark with large-scale test samples that represent diverse real-world scenes [Source: Single-Image Shadow Removal Using Deep Learning_ A Comprehensive Survey.pdf, chunk 4]. Additionally, existing datasets typically have low resolutions (e.g., 480 × 640), which is insufficient for real-world scenarios that often involve high-resolution imagery (e.g., 2k or 4k) [Source: Single-Image Shadow Removal Using Deep Learning_ A Comprehensive Survey.pdf, chunk 6].
*   **Shadow-agnostic regions**: Current networks are often incapable of accounting for the spatial correlation between remote areas, which is crucial for effectively de-shadowing images while preserving details in both shadowed and non-shadowed regions [Source: SpA-Former_An Effective and lightweight Transformer for image shadow removal.pdf, chunk 6].

This skill is based on a static snapshot of documents collected up to 2026-04-13.
It does not include papers published after that date. Coverage is focused on
the sub-topics present in the curated document set and may not represent the
entire CV/DL field.

---

## Example Q&A

Q: What are the key characteristics of a comprehensive dataset for evaluating shadow removal methods?
A: A comprehensive dataset for shadow removal should encompass large-scale test samples with a diverse range of scenes, including both indoor and outdoor environments. It must also feature challenging lighting conditions, such as various types of light sources and multiple shadows, to enable fair and rigorous evaluations. For instance, the NTIRE 2025 Image Shadow Removal Challenge uses a specific test split for evaluation. [Source: Single-Image Shadow Removal Using Deep Learning_ A Comprehensive Survey.pdf, chunk 179; NTIRE 2025 Image Shadow Removal Challenge Report.pdf, chunk 34]

Q: How are different shadow removal methods typically compared in research?
A: Shadow removal methods are typically compared through both quantitative and qualitative evaluations across various benchmarks. For example, specific models like DenseSR and DynRouteNet have undergone extensive cross-method comparison, with seven different models evaluated on challenging cases. Competitions, such as the NTIRE 2025 Image Shadow Removal Challenge, facilitate comparison by having multiple teams (e.g., 17 teams) submit solutions (e.g., FuShaRem, ACVLab, LUMOS, X-Shadow) for quantitative assessment. [Source: Single-Image Shadow Removal Using Deep Learning_ A Comprehensive Survey.pdf, chunk 18; DynRouteNet_Lightweight Facial Shadow Removal with Mask-Guided Dynamic Routing.pdf, chunk 77; NTIRE 2025 Image Shadow Removal Challenge Report.pdf, chunk 34, 2]

Q: What information is provided regarding loss function design in shadow removal research?
A: Loss functions are identified as a component of work architectures within deep learning for single-image shadow removal. However, the provided context does not contain detailed information regarding the specific design principles or types of loss functions utilized in this research area. [Source: Single-Image Shadow Removal Using Deep Learning_ A Comprehensive Survey.pdf, chunk 18]

Q: What evaluation metrics and procedures are used to assess shadow removal methods?
A: Evaluation metrics encompass quantitative evaluations and non-reference evaluation metrics. Additionally, user studies are employed to assess "Removal quality," which focuses on whether shadows are completely and cleanly eliminated without leaving residues or causing new artifacts. To ensure impartiality, such user studies often involve random shuffling of images and maintaining participant anonymity. Quantitative evaluations are, for example, a key part of challenges like the NTIRE 2025 Image Shadow Removal Challenge. [Source: Single-Image Shadow Removal Using Deep Learning_ A Comprehensive Survey.pdf, chunk 18, 179; DynRouteNet_Lightweight Facial Shadow Removal with Mask-Guided Dynamic Routing.pdf, chunk 80; NTIRE 2025 Image Shadow Removal Challenge Report.pdf, chunk 34]

Q: What specific considerations are given to handling different types of shadows, such as those from various light sources or multiple shadows?
A: While the context highlights the importance of incorporating "challenging lighting conditions such as various types of light sources and multiple shadows" into comprehensive benchmarks to advance the field, it does not explicitly detail specific methodologies or techniques developed to handle these particular shadow types. Generalized shadow removal is mentioned as a future research direction. [Source: Single-Image Shadow Removal Using Deep Learning_ A Comprehensive Survey.pdf, chunk 179, 18]

---

## Source References

| # | Filename | Type |
|---|----------|------|
| 1 | `Auto-Exposure Fusion for Single-Image Shadow Removal.pdf` | PDF |
| 2 | `Boundary-Aware Divide and Conquer A Diffusion-based Solution for Unsupervised Shadow Removal.pdf` | PDF |
| 3 | `DeS3_Adaptive Attention-driven Self and Soft Shadow Removal using ViT Similarity.pdf` | PDF |
| 4 | `DenseSR_Image Shadow Removal as Dense Prediction.pdf` | PDF |
| 5 | `Detail-Preserving Latent Diffusion for Stable Shadow Removal.pdf` | PDF |
| 6 | `Diff-Shadow_Global-guided Diffusion Model for Shadow Removal.pdf` | PDF |
| 7 | `DynRouteNet_Lightweight Facial Shadow Removal with Mask-Guided Dynamic Routing.pdf` | PDF |
| 8 | `Efficient Model-Driven Network for Shadow Removal.pdf` | PDF |
| 9 | `HomoFormer_Homogenized Transformer for Image Shadow Removal.pdf` | PDF |
| 10 | `Latent Feature-Guided Diffusion Models for Shadow Removal.pdf` | PDF |
| 11 | `Leveraging Inpainting for Single-Image Shadow Removal.pdf` | PDF |
| 12 | `NTIRE 2024 Image Shadow Removal Challenge Report.pdf` | PDF |
| 13 | `NTIRE 2025 Image Shadow Removal Challenge Report.pdf` | PDF |
| 14 | `OmniSR_Shadow Removal Under Direct and Indirect Lighting.pdf` | PDF |
| 15 | `PhaSR_Generalized Image Shadow Removal with Physically Aligned Priors.pdf` | PDF |
| 16 | `Polarization Guided Mask-Free Shadow Removal.pdf` | PDF |
| 17 | `Shadow Removal by a Lightness-Guided Network With Training on Unpaired Data.pdf` | PDF |
| 18 | `ShadowDiffusion_When Degradation Prior Meets Diffusion Model for Shadow Removal.pdf` | PDF |
| 19 | `ShadowFormer_Global Context Helps Shadow Removal.pdf` | PDF |
| 20 | `ShadowRefiner_Towards Mask-free Shadow Removal via Fast Fourier.pdf` | PDF |
| 21 | `Single-Image Shadow Removal Using Deep Learning_ A Comprehensive Survey.pdf` | PDF |
| 22 | `SoftShadow Leveraging Soft Masks for Penumbra-Aware Shadow Removal.pdf` | PDF |
| 23 | `SpA-Former_An Effective and lightweight Transformer for image shadow removal.pdf` | PDF |
| 24 | `Structure-Guided Diffusion Models for High-Fidelity Portrait Shadow Removal.pdf` | PDF |
| 25 | `WSRD_A Novel Benchmark for High Resolution Image Shadow Removal.pdf` | PDF |
| 26 | `paper_index.md` | MD |
