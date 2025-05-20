# Plant Seedlings Classifier using CNN

Plant Seedlings Classifier is the deep learning project for classifying 12 different plant seedlings using the Convolutional Neural Network (CNN). This project helped me learn the practical workflow of machine learning, including:

- Loading and preprocessing images with data augmentation
- Building a custom CNN model
- Training and validating the model
- Evaluating results using classification metrics
- Visualizing training history (accuracy & loss)
- Saving the trained model for future predictions

---

## **Dataset**

- **Name:** Plant Seedlings Classification  
- **Source:** [Kaggle – Plant Seedlings Classification](https://www.kaggle.com/c/plant-seedlings-classification/data)  
- The dataset contains images of **12 plant species** in separate folders.  
---

## **Technologies Used:**
- Python, TensorFlow, Keras
- NumPy, Matplotlib, Seaborn
- scikit-learn
---

## **Directory structure:**
```bash
plant-seedling-classifier/
│
├── data/
│   ├── train/ # subfolders for each plant species
│   └── test/ # test images
├── src/
│ ├── data_loader.py # loads and preprocesses images
│ ├── model.py # defines the CNN model
│ ├── train.py # trains the model
│ └── evaluate.py # evaluates and plots results
├── requirements.txt
└── README.md
```

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

## **Usage**
1. Train the Model
```bash
python src/train.py
```

- Trains the CNN model on the training data
- Saves the trained model as plant_seedlings_cnn.h5
- Plots training accuracy and loss

2. Evaluate the Model
```bash
python src/evaluate.py
```
- Loads the saved model
- Evaluates on the validation set
- Displays classification report and per-class accuracy

---
## **Results**
**Baseline CNN Performance**  
- Training accuracy reached ~60% after 20 epochs.  
- Validation/Test accuracy: ~11% (indicating overfitting).  
- Some classes (e.g., Loose Silky-bent, Scentless Mayweed) performed slightly better, but many species were misclassified due to visual similarity and limited dataset size.  

### Example Classification Report  

<table>
  <thead>
    <tr>
      <th>Class</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1-score</th>
      <th>Support</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Black-grass</td><td>0.00</td><td>0.00</td><td>0.00</td><td>52</td></tr>
    <tr><td>Charlock</td><td>0.12</td><td>0.12</td><td>0.12</td><td>78</td></tr>
    <tr><td>Cleavers</td><td>0.09</td><td>0.12</td><td>0.11</td><td>57</td></tr>
    <tr><td>Common Chickweed</td><td>0.11</td><td>0.14</td><td>0.12</td><td>122</td></tr>
    <tr><td>Common wheat</td><td>0.05</td><td>0.05</td><td>0.05</td><td>44</td></tr>
    <tr><td>Fat Hen</td><td>0.08</td><td>0.06</td><td>0.07</td><td>95</td></tr>
    <tr><td>Loose Silky-bent</td><td>0.13</td><td>0.19</td><td>0.16</td><td>130</td></tr>
    <tr><td>Maize</td><td>0.00</td><td>0.00</td><td>0.00</td><td>44</td></tr>
    <tr><td>Scentless Mayweed</td><td>0.14</td><td>0.14</td><td>0.14</td><td>103</td></tr>
    <tr><td>Shepherds Purse</td><td>0.00</td><td>0.00</td><td>0.00</td><td>46</td></tr>
    <tr><td>Small-flowered Cranesbill</td><td>0.13</td><td>0.16</td><td>0.14</td><td>99</td></tr>
    <tr><td>Sugar beet</td><td>0.07</td><td>0.08</td><td>0.07</td><td>77</td></tr>
  </tbody>
</table>

**Overall metrics**  
- Accuracy: **0.11**  
- Macro avg: Precision = 0.08, Recall = 0.09, F1 = 0.08  
- Weighted avg: Precision = 0.09, Recall = 0.11, F1 = 0.10  
---

## **Improvements & Future Work**  

While the baseline CNN struggled, this project helped me understand the **end-to-end ML workflow** (data preprocessing, model training, evaluation).  

For future improvement:  
- Apply **Transfer Learning** with pre-trained networks (e.g., ResNet, MobileNet, EfficientNet) to boost performance.  
- Use **class weights or oversampling** to handle class imbalance.  