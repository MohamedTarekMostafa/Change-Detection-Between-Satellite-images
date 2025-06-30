# Change Detection in Satellite Imagery using U-Net

This repository contains the complete implementation of a **Change Detection** model using a **U-Net** architecture. The goal is to detect changes between pairs of satellite images taken at different times of the same geographic location.

##  Project Info
- **Author**: Mohamed Tarek Mostafa
- **Graduation Project**: This project is part of my final year graduation project. I was responsible for the full implementation.
- **Model**: U-Net
- **Dataset**: LEVIR-CD
- **Test Accuracy**: 97%
- **Validation Dice Coefficient**: 0.7481
- **Manual Loss**: 0.1

##  Dataset
- Dataset used: [LEVIR-CD](https://drive.google.com/drive/folders/1X6oA_FoKCZrdQSQZW1iS1mAbiPUwtKNF?usp=sharing)
- Images are 1024×1024 and were resized to 256×256 during preprocessing.

##  Model Architecture
The U-Net model was used to predict binary change masks between pre- and post-event satellite images.

##  Directory Structure
```
├── data/
│   ├── train_images.npy
│   ├── train_masks.npy
│   ├── val_images.npy
│   ├── val_masks.npy
│   ├── test_images.npy
│   └── test_masks.npy
│
├── models/
│   └── unet_model.h5
│
├── results/
│   └── sample_prediction.png
│
├── src/
│   ├── model.py         # U-Net architecture
│   └── losses.py        # Custom loss functions
│
├── Change_Detection_Main_Code.ipynb  # Training notebook
├── Testing_Model_Code.ipynb          # Evaluation notebook
├── train.py                          # Training script
├── test.py                           # Testing script
├── requirements.txt
└── README.md
```

##  How to Run

### 1. Clone the repository
```bash
git clone https://github.com/MohamedTarekMostafa/Change-Detection-Unet
cd Change-Detection-Unet
```

### 2. Setup environment
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python train.py
```

### 4. Test the model
```bash
python test.py
```

##  Sample Results
A sample prediction is saved in `results/sample_prediction.png` after running `test.py`.

##  Contact
- **Email**: mohamedelgebaly921@gmail.com
- **LinkedIn**: [Mohamed Tarek Mostafa](https://www.linkedin.com/in/mohamed-tarek-mostafa-936452221/)
- **GitHub**: [MohamedTarekMostafa](https://github.com/MohamedTarekMostafa)

---
> *"Detecting what changes, changes everything."* 
