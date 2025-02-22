# GPU Setup Guide for Kaggle Notebooks
This guide explains how to properly import and run notebooks with GPU support on Kaggle, specifically focusing on accessing 2xT4 GPUs which are essential for the Reinforcement Learning in PostTraining and Instruction Fine Tuning modules.

## 📥 Prerequisites: Clone the Repository
Before starting, you need to clone the workshop repository to your local machine:

```
git clone https://github.com/thomsonreuters/labs_AMLD25_Workshop.git
```

This repository contains all the necessary notebooks and materials for the workshop. After cloning, you'll need to follow the instructions below to set up the proper GPU environment in Kaggle.

## 🔑 Important Notes
* This solution ensures your credentials remain private (no connector creation in GitHub)
* Provides access to 2 GPUs (2xT4), which is required for the final module
* If you have access to more powerful GPUs (2xA10 or better), you can use your own provider

## 🚀 Setup Instructions
### 1. Import Notebooks to Kaggle
1. Go to [Kaggle](https://www.kaggle.com/)
2. Click on Code → ``+ New Notebook``
3. Click on ``File`` tab → ``Import Notebook``
4. Upload the notebook with first exercise from the TR Labs repo cloned to our laptop

### 2. Configure GPU Runtime
1. ``Setting`` → ``Accelerator`` → ``GPU T4 x 2``, then ``Turn on`` 
2. Verify GPU Configuration
Run this code cell to confirm you have access to 2 GPUs:
```{bash}
!nvidia-smi
```
Expected output should show 2 T4 GPUs.

3. (Optional) Alternative Setups
If you have access to more powerful GPUs:

* 2xA10 or better: You can use your own cloud provider
   * Local setup: Ensure you have equivalent or better GPU power than 2xT4

## ⚡ Performance Requirements
Minimum GPU requirements for the Reinforcement Learning module:

2x NVIDIA T4 GPUs or equivalent
CUDA compatibility

## 🔍 Troubleshooting
If you don't see 2 GPUs:

1. Check your Notebook Settings
2. Ensure you have selected GPU 2xT4
3. Restart the kernel if needed
4. Check your weekly GPU usage quota in Kaggle

If you can't select an instance with GPUs:
1. GPU accelerators are only available for verified users
2. Go to your Kaggle profile (top right profile icon -> click on "Your Profile")
3. Click on "Settings" -> "Phone verification".
4. Follow the steps using a valid phone number.

## ⚠️ Important Reminders
1. Kaggle GPU Quotas:
    * Standard users: 30 hours/week of GPU usage
    * Check your remaining hours in Account Settings
2. Notebooks auto-save
3. Sessions timeout after inactivity
4. Don't share API keys or credentials in notebooks

## 📝 Additional Notes
1. Kaggle's GPU sessions last up to 12 hours
2. All packages are pre-installed in Kaggle environment
3. Internet access is enabled by default
4. Use !pip install for additional packages

## 🔒 Security
This setup ensures:

* No credentials are stored in GitHub
* Secure GPU access through Kaggle's infrastructure
* Protected notebook environment

## 👥 Contributors
Giofré, Daniele (@dgiofre) and Gysel, Philipp (@PhilippGyselTR) and Rudnikowicz, Bartosz (@bartoszrud) and Taneva-Popova, Bilyana (@BilyanaTanevaTR) and Trautmann, Dietrich (@DietrichTrautmannTR)


