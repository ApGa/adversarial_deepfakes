# adversarial_deepfakes
Deepfakes with an adversarial twist.

This repository provides code and additional materials for the paper:
"Adversarial perturbations fool deepfake detectors", Apurva Gandhi and Shomik Jain, Submitted to IJCNN 2020.

The paper uses adversarial perturbations to enhance deepfake images and fool common deepfake detectors. We also explore two improvements to deepfake detectors: (i) Lipshitz regularization, and (ii) Deep Image Prior (DIP). 

## Files:
- adv_examples.py: Adversarial Examples Creation
- classifier.py: Deepfake Detector Creation 
- cw.py: Carlini-Wagner L2 Norm Attack
- dip_template.py: Deep Image Prior Defense
- evaluation.py: Model Evaluation Script
- generate_dataset.py: Deepfake Generation Script
