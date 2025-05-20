# Plant Seedlings Classifier using CNN

SeedlingNet is a deep learning project for **classifying 12 different plant seedlings** using a Convolutional Neural Network (CNN). The project demonstrates an end-to-end workflow:

- Loading and preprocessing images with data augmentation  
- Building a custom CNN model  
- Training and validating the model  
- Evaluating results using confusion matrix and classification report  
- Visualizing training history (accuracy & loss)  
- Saving the trained model for future predictions  

---

## **Dataset**

- **Name:** Plant Seedlings Classification  
- **Source:** [Kaggle – Plant Seedlings Classification](https://www.kaggle.com/c/plant-seedlings-classification/data)  
- The dataset contains images of **12 plant species** in separate folders.  
---

### **Technologies Used:**
-Python, TensorFlow, Keras
- NumPy, Matplotlib, Seaborn
- scikit-learn
---

**Directory structure:**
data/
├── train/ # subfolders for each plant species
└── test/ # test images
├── src/
│ ├── data_loader.py # loads and preprocesses images
│ ├── model.py # defines the CNN model
│ ├── train.py # trains the model
│ └── evaluate.py # evaluates and plots results
├── requirements.txt
└── README.md


---

## **Installation**

1. Clone the repo:
```bash
git clone https://github.com/JaskiratCodeKaur/plant-seedling-classifier
cd plant-seedling-classifier
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
---

### **Usage**
1. Train the Model
```bash
python src/train.py
```

- Trains the CNN model on the training data
- Saves the trained model as plant_seedlings_cnn.h5
- Plots training/validation accuracy and loss

2. Evaluate the Model
```bash
python src/evaluate.py
```
- Loads the saved model
- Evaluates on the validation set
- Displays confusion matrix and classification report

---
### **Results**
