# Helmet Compliance Detection Using CNNs and Transfer Learning

## Project Overview

Workplace safety is critical in hazardous environments such as construction sites, factories, and industrial facilities.

One of the most important safety requirements is the use of protective helmets. Failure to wear helmets can significantly increase the risk of serious head injuries caused by falling objects, machinery, or other workplace hazards.

Manual safety monitoring can become difficult and inefficient in large-scale environments. This project develops a **computer vision classification system** for SafeGuard Corp that automatically determines whether a worker is:

- **With Helmet**
- **Without Helmet**

The project compares a custom Convolutional Neural Network with multiple VGG16-based transfer-learning approaches to identify the model best suited for helmet compliance monitoring.

---

## Business Objective

The objective of this project is to help SafeGuard Corp:

- Automatically identify workers who are not wearing safety helmets
- Improve workplace safety compliance
- Reduce dependence on manual monitoring
- Detect unsafe behavior more consistently
- Support scalable monitoring across industrial locations
- Evaluate CNN and transfer-learning approaches for safety image classification
- Select a model that prioritizes detection of unsafe workers

Because failing to identify a worker without a helmet represents the greatest safety risk, **Recall for the `Without Helmet` class is especially important**.

---

## Machine Learning Problem

This is a **binary image classification problem**.

The two target classes are:

- `With Helmet`
- `Without Helmet`

The model receives an image of a worker and predicts whether the worker is wearing appropriate helmet protection.

```text
Worker Image
     ↓
Image Classification Model
     ↓
With Helmet / Without Helmet
```

---

## Dataset

The dataset contains **4,125 images** collected from workplace environments.

### Class Distribution

| Class | Number of Images |
|---|---:|
| With Helmet | 3,161 |
| Without Helmet | 964 |
| **Total** | **4,125** |

The dataset is therefore imbalanced, with substantially more examples of workers wearing helmets than workers without helmets.

### Dataset Characteristics

Images contain variations in:

- Lighting
- Camera angle
- Worker posture
- Workplace environment
- Background
- Worker activity

Workers are shown in environments such as:

- Construction sites
- Factories
- Industrial facilities

and may be:

- Standing
- Walking
- Using tools
- Performing work activities

These variations help evaluate how well the models learn helmet-related visual patterns across different conditions.

---

## Project Workflow

```text
Image Dataset
      ↓
Data Exploration
      ↓
Class Distribution Analysis
      ↓
Train / Validation / Test Split
      ↓
Image Normalization
      ↓
Custom CNN
      ↓
VGG16 Transfer Learning
      ↓
VGG16 + FFNN
      ↓
VGG16 + FFNN + Augmentation
      ↓
Model Comparison
      ↓
Final Model Selection
      ↓
Business Recommendations
```

---

# Exploratory Data Analysis

## Data Overview

The image dataset was loaded and inspected to understand:

- Dataset dimensions
- Image shapes
- Class labels
- Number of observations
- Class distribution

---

## Image Visualization

Random samples from both classes were visualized along with their labels.

This helped confirm that the images represent diverse workplace conditions and provided an initial understanding of the visual differences between helmet and non-helmet examples.

---

## Class Imbalance

The dataset contains:

```text
With Helmet    → 3,161 images
Without Helmet →   964 images
```

The `Without Helmet` class is substantially smaller.

This imbalance is important because a model can achieve high overall accuracy while still performing poorly on the smaller but safety-critical class.

For this reason, model evaluation focuses not only on accuracy but also on **Recall for workers without helmets**.

---

# Data Preprocessing

## Train / Validation / Test Split

The dataset was divided into:

- Training data
- Validation data
- Test data

The training set was used for model learning, the validation set for model comparison, and the test set for final evaluation.

---

## Image Normalization

Pixel values were normalized before training.

Original image pixel values typically fall within:

```text
0 – 255
```

Normalization transforms them into a smaller numerical range:

```text
0 – 1
```

This helps improve neural network training stability and convergence.

---

# Model 1 — Convolutional Neural Network From Scratch

A custom **Convolutional Neural Network (CNN)** was developed as the first image classification model.

CNNs are especially effective for image data because they automatically learn visual features such as:

- Edges
- Shapes
- Patterns
- Textures
- Spatial relationships

A simplified CNN workflow is:

```text
Input Image
     ↓
Convolution
     ↓
Feature Maps
     ↓
Pooling
     ↓
Additional Convolution Layers
     ↓
Flatten / Dense Layers
     ↓
Helmet Classification
```

The model configuration included:

- Loss function
- Optimizer
- Batch size
- Number of epochs

The model was trained and evaluated using training and validation data.

---

# Model 2 — VGG16 Transfer Learning

The second approach used **VGG16**, a pretrained convolutional neural network.

VGG16 was originally trained on the ImageNet dataset and has learned generic visual features such as:

- Edges
- Shapes
- Textures
- Object structures

Transfer learning allows these learned representations to be reused for the helmet classification problem.

```text
Image
   ↓
Pretrained VGG16
   ↓
Extracted Image Features
   ↓
Helmet Classification
```

The pretrained convolutional base was used to extract image features for classification.

---

# Model 3 — VGG16 + Feed-Forward Neural Network

A custom feed-forward neural network was added on top of the VGG16 feature extractor.

```text
Input Image
      ↓
VGG16 Feature Extractor
      ↓
Extracted Features
      ↓
Custom Dense Layers
      ↓
Binary Classification
```

This allows the pretrained network to provide general image representations while the custom classification layers learn patterns specific to helmet compliance.

---

# Model 4 — VGG16 + FFNN + Data Augmentation

The VGG16-based architecture was also trained using **data augmentation**.

Data augmentation creates modified versions of training images to expose the model to greater visual diversity.

Augmentation can include transformations such as:

- Rotation
- Translation
- Zoom
- Flipping
- Other image transformations

Conceptually:

```text
Original Image
      ↓
Image Augmentation
      ↓
Multiple Training Variations
      ↓
VGG16
      ↓
Classification Model
```

The objective is to improve model generalization across different viewing conditions.

---

# Model Comparison

The models were compared based on their ability to correctly classify both helmet and non-helmet cases.

The evaluated approaches included:

| Model | Approach |
|---|---|
| Custom CNN | CNN built from scratch |
| VGG16 Base | Transfer learning |
| VGG16 + FFNN | Transfer learning with custom classification layers |
| VGG16 + FFNN + Augmentation | Transfer learning with augmented training data |

Model selection considered:

- Accuracy
- Precision
- Recall
- Class-specific performance
- Generalization
- Ability to detect workers without helmets

---

## Key Model Performance Insights

### Custom CNN

The custom CNN achieved the strongest performance for detecting the safety-critical `Without Helmet` class.

Its strong Recall means that fewer unsafe workers are missed.

### VGG16 Models

The VGG16-based models achieved strong overall classification performance but showed lower effectiveness in identifying the minority `Without Helmet` class.

### Data Augmentation

Data augmentation improved generalization but did not significantly improve Recall for the safety-critical minority class.

---

# Final Model Selection

## Custom CNN

The **custom CNN was selected as the final model**.

The primary reasons were:

- Highest effectiveness in identifying workers without helmets
- Strong Recall for the safety-critical class
- Good generalization
- Efficient architecture
- Strong overall classification performance

For this business problem, correctly detecting unsafe workers is more important than maximizing overall accuracy.

---

# Why Recall Matters

Consider a worker who is not wearing a helmet.

### Correct Detection

```text
No Helmet
    ↓
Model Detects Violation
    ↓
Safety Alert
    ↓
Supervisor Intervention
```

This helps prevent a potential safety incident.

### Missed Detection

```text
No Helmet
    ↓
Model Predicts Helmet
    ↓
No Alert
    ↓
Worker Remains at Risk
```

A false negative therefore represents the most serious model error.

This is why **Recall for the `Without Helmet` class is prioritized**.

---

# Key Insights

## 1. CNNs Can Effectively Detect Helmet Compliance

The custom CNN successfully learned visual patterns associated with helmet usage.

The results demonstrate that image classification can automate an important workplace safety monitoring task.

---

## 2. Safety Metrics Matter More Than Overall Accuracy

A model may achieve strong accuracy while still missing unsafe workers because the dataset contains more helmet-compliant examples.

For SafeGuard Corp, identifying workers without helmets is more important than maximizing overall accuracy.

---

## 3. Class Imbalance Affects Failure Detection

The dataset contains significantly fewer `Without Helmet` images.

This affects how well models learn minority-class patterns and reinforces the importance of evaluating class-specific metrics.

---

## 4. Simpler Models Can Outperform More Complex Architectures

The custom CNN performed better on the safety-critical metric than the more complex pretrained VGG16 approaches.

This demonstrates that model complexity does not automatically translate into better business performance.

---

## 5. Structural Image Features Are Important

Helmet detection relies strongly on visual structures such as:

- Shape
- Edges
- Head-region patterns
- Object structure

These features can be effectively learned by convolutional architectures.

---

# Business Impact

## Automated Safety Monitoring

The model enables automated analysis of worker images instead of relying entirely on manual observation.

This can reduce monitoring workload and improve scalability.

---

## Reduced Safety Risk

High Recall for workers without helmets allows unsafe situations to be identified earlier.

Earlier intervention can reduce the likelihood of workplace injuries.

---

## Operational Efficiency

Automated monitoring enables safety teams to supervise larger areas without proportionally increasing manual monitoring effort.

---

## CCTV Integration

The image classification model can support helmet-monitoring workflows using images captured from existing surveillance systems.

A high-level workflow could be:

```text
Workplace Camera
      ↓
Worker Image
      ↓
Helmet Classification
      ↓
Compliance Status
      ↓
Safety Monitoring System
```

---

# Business Recommendations

## Recommendation 1 — Use the Custom CNN for Helmet Classification

The custom CNN demonstrated the strongest ability to identify workers without helmets and is therefore the preferred model for this safety use case.

---

## Recommendation 2 — Prioritize the `Without Helmet` Recall Metric

Performance monitoring should emphasize:

- Recall for `Without Helmet`
- False Negative Rate
- Precision
- Overall Accuracy

The most important objective is minimizing unsafe workers who are incorrectly classified as compliant.

---

## Recommendation 3 — Integrate Predictions With Safety Monitoring

Model predictions can support automated compliance workflows.

```text
Worker Image
      ↓
CNN
      ↓
Helmet / No Helmet
      ↓
Compliance Monitoring
      ↓
Safety Response
```

This can help safety teams identify potentially unsafe situations more efficiently.

---

## Recommendation 4 — Use Automated Monitoring to Support Human Oversight

The system can help safety officers focus attention on images or areas where violations are detected.

AI-based monitoring can complement existing safety supervision rather than replacing human safety oversight.

---

# Additional Business Applications

Helmet compliance predictions can support:

- Workplace safety monitoring
- Compliance reporting
- Safety audits
- Violation tracking
- Identification of frequently unsafe areas
- Analysis of compliance trends
- Safety-management dashboards

The system can also provide data that helps management understand where helmet violations occur most frequently.

---

# Executive Summary

A computer vision system was developed to automatically classify workplace images into:

- **With Helmet**
- **Without Helmet**

The project evaluated four deep learning approaches:

1. Custom CNN
2. VGG16 transfer learning
3. VGG16 with custom feed-forward layers
4. VGG16 with custom layers and data augmentation

The dataset contained **4,125 workplace images**, including **3,161 helmet-compliant images and 964 non-compliant images**.

Because workers without helmets represent the smaller but safety-critical class, model selection emphasized **Recall for the `Without Helmet` class** rather than overall accuracy alone.

The **custom CNN was selected as the final model** because it provided the strongest ability to detect workers without helmets while maintaining strong overall performance and generalization.

The project demonstrates how computer vision and deep learning can support scalable workplace safety monitoring and automate helmet-compliance detection.

---

# Key Concepts Demonstrated

This project demonstrates practical application of:

- Computer Vision
- Image Classification
- Deep Learning
- Convolutional Neural Networks
- Transfer Learning
- VGG16
- Feed-Forward Neural Networks
- Image Normalization
- Train / Validation / Test Splitting
- Data Augmentation
- Binary Classification
- Class Imbalance
- Accuracy
- Precision
- Recall
- F1 Score
- False Negatives
- Model Comparison
- Model Selection
- Workplace Safety AI
- Safety-Critical Model Evaluation

---

# Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- TensorFlow
- Keras
- VGG16
- Scikit-learn
- Google Colab
- Jupyter Notebook

---

# Repository Structure

```text
helmet-compliance-detection-cnn/
│
├── README.md
├── Helmet_Detection_Notebook.ipynb
├── Helmet_Detection_Notebook.html
│
└── data/
    ├── images.npy
    └── labels.csv
```

## Files

- README.md — Project overview, modeling approach, model comparison, and business insights

- Helmet_Detection_Notebook.ipynb — Complete Google Colab notebook containing EDA, preprocessing, CNN development, transfer learning, evaluation, and recommendations

- Helmet_Detection_Notebook.html — Rendered version of the notebook for convenient viewing

- data/images.npy — NumPy array containing the image data used for model training and evaluation

- data/labels.csv — Labels corresponding to the images, identifying each sample as With Helmet or Without Helmet
---

# Conclusion

This project demonstrates an end-to-end **computer vision and deep learning solution for workplace helmet compliance detection**.

A custom CNN and multiple VGG16 transfer-learning approaches were developed and compared.

The final **custom CNN** provided the strongest performance on the safety-critical `Without Helmet` class, making it the most suitable model among those evaluated for SafeGuard Corp's helmet-monitoring use case.

The project demonstrates an important principle in safety-critical machine learning:

> The most useful model is not necessarily the one with the highest overall accuracy, but the one that best identifies the events carrying the greatest real-world risk.

By automatically identifying helmet violations, computer vision can support workplace safety teams, improve monitoring efficiency, and help organizations maintain stronger safety compliance.