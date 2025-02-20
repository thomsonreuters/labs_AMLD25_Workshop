# Reinforcement Learning Workshop Module
This folder contains hands-on exercises for learning different preference learning techniques using transformer models. 
All notebooks in ``kaggle`` folder are designed to run on Kaggle.

## Prerequisites
* Kaggle account
* Basic understanding of transformers and PyTorch
* Follow the environment setup instructions in the root directory's **README**
## Exercises Overview
### Exercise 1: Direct Preference Optimization (DPO) Training
* Introduction to DPO
* Basic implementation of DPO training
* Understanding preference learning fundamentals
* Noisy training signals
* Suboptimal performance
* Inefficient learning

File: ex1_dpo_training.ipynb
### Exercise 2: Mixed Collections DPO Training
* Working with multiple data collections
* Maintain general knowledge
* Preserve essential capabilities
* Balance new and existing skills

File: ex2_mixingcollections_dpo_training.ipynb
### Exercise 3: Regularized Preference Optimization (RPO) Training
* Understanding RPO methodology
* Implementation of RPO training
* Comparing RPO with DPO

File: ex3_rpo_training.ipynb
### Exercise 4: Online DPO Training (Optional)
* Real-time preference learning
* Online learning implementation
* Reward Model versus LLM-as-Judge
* Advanced concepts in DPO

File: ex4_onlinedpo_training.ipynb
### Getting Started
1. Make sure you have completed the environment setup as described in the root README
2. Open the exercises in order (1-3, 4 is optional)
3. Each notebook contains:
  * Description of the exercise and pseudo-solution
