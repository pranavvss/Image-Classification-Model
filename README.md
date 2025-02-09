# Tomato Classification Model

A deep learning project that classifies tomatoes as either **ripe** or **unripe** using computer vision techniques. This project was developed to automate the sorting process in agricultural settings, inspired by a hypothetical business scenario for **FreshHarvest Inc.**. By leveraging Convolutional Neural Networks (CNNs), transfer learning (using pre-trained ResNet models), data augmentation, and interpretability tools like Integrated Gradients, the model aims to improve the quality control of tomato produce.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Business Problem & Motivation](#business-problem--motivation)
- [Dataset and Data Preparation](#dataset-and-data-preparation)
  - [Data Collection](#data-collection)
  - [Data Wrangling and Exploratory Data Analysis](#data-wrangling-and-exploratory-data-analysis)
  - [Data Preprocessing](#data-preprocessing)
- [Model Architecture and Training](#model-architecture-and-training)
  - [Vanilla CNN Model](#vanilla-cnn-model)
  - [Training Loop and TensorBoard Monitoring](#training-loop-and-tensorboard-monitoring)
  - [Transfer Learning with ResNet50](#transfer-learning-with-resnet50)
- [Model Interpretability](#model-interpretability)
- [Quantized Models & Efficiency](#quantized-models--efficiency)
- [Semantic Segmentation Example](#semantic-segmentation-example)
- [Results and Outcome](#results-and-outcome)
- [Future Work](#future-work)
- [Additional Resources](#additional-resources)
- [Requirements](#requirements)
- [Dataset Access](#dataset-access)
- [Grayscale Usage](#grayscale-usage)
- [Conclusion](#conclusion)

---

## Project Overview

This project presents a complete pipeline for classifying tomatoes by ripeness using deep learning. The model is developed in Google Colab and integrates several key steps:

- **Data Ingestion and Preprocessing:** Reading images and labels from a Kaggle dataset.
- **Model Development:** Building a CNN from scratch and fine-tuning a pre-trained ResNet50.
- **Model Interpretability:** Using Integrated Gradients from the [Captum](https://captum.ai/) library to understand feature attributions.
- **Quantization:** Discussing both pre-training and post-training quantization to improve efficiency on resource-constrained devices.
- **Visualization:** Monitoring training progress and embedding visualizations with TensorBoard.

An example image of the project workflow is shown below:

<img src="https://github.com/user-attachments/assets/9b8a25a4-6c43-428a-86c5-be83a97f235f" alt="Workflow Diagram" style="width:550px;"/>

---

## Business Problem & Motivation

Modern agricultural practices increasingly rely on automation to enhance crop quality and yield. **FreshHarvest Inc.**—a hypothetical agricultural technology company—is looking to replace manual tomato sorting with an automated system. Manual sorting is:
- **Time-consuming**
- **Labor-intensive**
- **Prone to human error**

By deploying a robust tomato classification model, the company aims to:
- Reduce labor costs.
- Increase sorting speed.
- Improve the overall quality of tomatoes sent to market.

---

## Dataset and Data Preparation

### Data Collection

The dataset comprises **177 images** of tomatoes along with corresponding label files indicating whether a tomato is ripe or unripe. The data was sourced from [Kaggle](https://www.kaggle.com/datasets/sumn2u/riped-and-unriped-tomato-dataset).

### Data Wrangling and Exploratory Data Analysis

- **Data Wrangling:**  
  - Removed corrupted images.
  - Ensured accurate matching between images and labels.
  - Verified image quality.

- **Exploratory Data Analysis (EDA):**  
  - Visualized the distribution of classes (ripe vs. unripe) to check for imbalance.
  - Displayed sample images along with their labels.

Example code to visualize an image sample:

```python
import matplotlib.pyplot as plt
from PIL import Image

test_img = Image.open(first_sample)
plt.imshow(np.asarray(test_img))
plt.show()
```

### Data Preprocessing

Preprocessing steps are critical for model performance:
- **Resizing:** Images are resized to 224x224 pixels.
- **Normalization:** Images are normalized using ImageNet’s mean and standard deviation values.
- **Data Augmentation:** Techniques such as rotation, flipping, zooming, and shifting are applied to artificially increase dataset diversity.

A typical transformation pipeline is defined as:

```python
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

---

## Model Architecture and Training

### Vanilla CNN Model

The project begins with a vanilla CNN developed from scratch. This network includes:
- Two convolutional layers with ReLU activations and max pooling.
- Fully connected layers that output a binary prediction (ripe or unripe).

This simple model serves as a foundation before exploring more advanced methods like transfer learning.

### Training Loop and TensorBoard Monitoring

The training loop involves:
- **Loss Function:** Binary Cross-Entropy with logits.
- **Optimizer:** Adam, with learning rate adjustments based on training dynamics.
- **TensorBoard:** Used to visualize training and validation loss metrics.

An excerpt from the training loop:

```python
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_dataloader):
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
```

TensorBoard logging example:

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/tomato_classification')
writer.add_scalars('Training vs. Validation Loss',
                   {'Training': avg_loss, 'Validation': avg_vloss},
                   epoch * len(train_dataloader) + i)
```

### Transfer Learning with ResNet50

To improve performance and speed up training, the project leverages transfer learning with ResNet50.

```python
import torch.nn as nn
import torchvision.models as models

model = models.resnet50(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True
```

---

## Model Interpretability

To interpret model decisions, we use **Integrated Gradients** from Captum.

```python
from captum.attr import IntegratedGradients

integrated_gradients = IntegratedGradients(model)
attributions_ig = integrated_gradients.attribute(image, target=label, n_steps=200)
```

### Quantized Models & Efficiency

Quantization reduces model size and speeds up inference by converting 32-bit floating point weights to lower-precision formats (e.g., 8-bit integers). Two strategies are covered:
	•	Post-training Quantization (PTQ): Quantizes a trained model.
	•	Quantization-Aware Training (QAT): Trains the model while simulating quantization effects.

Important concepts:
	•	Symmetric vs. Asymmetric Quantization:
	•	Symmetric: Zero point is typically zero.
	•	Asymmetric: Zero point is calculated based on data distribution.
	•	Fake Quantization: Used in QAT to simulate quantization effects during training.

Code snippet for tensor quantization:

```python
quint8_tensor = torch.quantize_per_tensor(preprocessed_img, scale=1.0, zero_point=0, dtype=torch.quint8)
dequantized_tensor = quint8_tensor.dequantize()
```

## Semantic Segmentation Example

Beyond classification, the project demonstrates semantic segmentation using FCN ResNet50. This model labels each pixel in an image, with the output visualized as a segmentation mask.

Example code:

```python
model = fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT)
with torch.no_grad():
    prediction = model(batch)["out"]
mask = prediction.softmax(dim=1)[0, class_to_idx["dog"]]
to_pil_image(mask).show()
```
This example shows how similar techniques can be applied to more complex computer vision tasks.

## Results and Outcome
	•	Training Outcome:
	•	Training and validation losses steadily decreased, demonstrating effective learning.
	•	Accuracy: The models achieved up to 83% accuracy on the validation set.
	•	Although the goal was to reach above 90%, the achieved accuracy is promising given the dataset size and model complexity.
	•	Visualization:
	•	TensorBoard was used to monitor the training process, providing valuable insights into model performance.

Example loss curve visualization:

![image](https://github.com/user-attachments/assets/a1ba4cba-acd9-4535-b019-c9ee68125bb9)

## Future Work

Potential improvements include:
	•	Expanding the Dataset: A larger, more diverse dataset may improve model generalization.
	•	Robotic Integration: Combining the model with AI-driven robots using OpenCV for real-time image processing and control.
	•	Advanced Architectures: Experimenting with state-of-the-art models and further fine-tuning with quantization-aware training (QAT) for deployment on edge devices.
	•	Deployment: Developing a prototype or web service to demonstrate real-time tomato sorting.

## Additional Resources

To deepen your understanding, consider exploring these resources:
	•	Books:
	•	[Python Data Science Handbook: Essential Tools For Working With Data by Jake VanderPlas](https://github.com/jakevdp/PythonDataScienceHandbook)
	•	Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow 2 by Aurelien Geron

(The second part of this book covers advanced topics like RNNs, CNNs, and deep neural networks.)
	•	Research Papers:
	•	[Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
	•	Libraries and Frameworks:
	•	[PyTorch](https://pytorch.org)
	•	TorchVision
	•	[Captum](https://captum.ai)

## Requirements

To run this project, ensure that your environment meets the following requirements:
	•	Python: 3.6 or higher
	•	PyTorch: 1.7.0 or higher
	•	TorchVision: 0.8.1 or higher
	•	Captum: 0.4.0 or higher
	•	Pandas: 1.1.5 or higher
	•	Matplotlib: 3.3.2 or higher
	•	Pillow: 7.2.0 or higher
	•	TensorBoard: 2.3.0 or higher
	•	Google Colab: Recommended for running the notebook

Install the required packages with:

```
pip install torch torchvision captum matplotlib tensorboard pandas pillow
```

## Dataset Access

Due to file size constraints, the dataset is not included in this repository. You can download the dataset from Kaggle:
	•	[Riped and Unriped Tomato Dataset](https://www.kaggle.com/datasets/sumn2u/riped-and-unriped-tomato-dataset)

## Grayscale Usage

Converting images to grayscale reduces complexity by focusing on intensity rather than color. The grayscale conversion formula is:

[
\text{Gray} = 0.2989 \times R + 0.5870 \times G + 0.1140 \times B
]

Grayscale image examples:

![image](https://github.com/user-attachments/assets/05ac0868-7322-4267-abe3-6fba4e5e2b63)
![image](https://github.com/user-attachments/assets/8bf10103-3e32-4cc9-9e64-3341a17a85a2)

## Conclusion

This project demonstrates a comprehensive approach to automating tomato sorting in agricultural settings. By combining classical CNN architectures with transfer learning, interpretability methods, and quantization techniques, we have built a system capable of distinguishing between ripe and unripe tomatoes. Although the current model achieves 83% accuracy, further data collection, experimentation, and integration with hardware could push performance even higher.

At a larger scale, it is entirely feasible to integrate this tomato classification model with AI-driven robotics and real-time image processing using OpenCV.

# Happy coding and happy farming!
