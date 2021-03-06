# adversarial_deepfakes
Deepfakes with an adversarial twist.

This repository provides code and additional materials for the paper:
"Adversarial perturbations fool deepfake detectors", Apurva Gandhi and Shomik Jain, To Appear in IJCNN 2020.

The paper uses adversarial perturbations to enhance deepfake images and fool common deepfake detectors. We also explore two improvements to deepfake detectors: (i) Lipschitz regularization, and (ii) Deep Image Prior (DIP). 

Link to preprint: https://arxiv.org/abs/2003.10596.

## Files:
- adv_examples.py: Adversarial Examples Creation
- classifier.py: Deepfake Detector Creation 
- cw.py: Carlini-Wagner L2 Norm Attack
- dip_template.py: Deep Image Prior Defense
- evaluation.py: Model Evaluation Script
- generate_dataset.py: Deepfake Generation Script
- ijcnn_presentation.pdf: Presentation Slides from IJCNN 2020

## References:
- Deepfake Literature Review: https://arxiv.org/pdf/1909.11573.pdf
- Deepfake Creation Implementation: https://github.com/shaoanlu/fewshot-face-translation-GAN
- Fast Gradient Sign Method Paper: https://arxiv.org/abs/1412.6572
- Carlini-Wagner Attack Paper: https://arxiv.org/abs/1608.04644
- Carlini-Wagner Attack Implementation: https://github.com/kkew3/pytorch-cw2
- Lipshitz Regularization Paper: https://www.nature.com/articles/s42256-019-0104-6
- Deep Image Prior Paper: https://arxiv.org/abs/1711.10925
- Deep Image Prior Implementation: https://github.com/DmitryUlyanov/deep-image-prior
