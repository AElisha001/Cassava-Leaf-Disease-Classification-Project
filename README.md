Cassava Leaf Disease Classification Project
This repository contains the source code for a computer vision project aimed at classifying cassava leaf diseases using deep learning techniques. The project leverages transfer learning with a pre-trained ResNet50 model and is optimized to run on Tensor Processing Units (TPUs) provided by Kaggle.

📌 Project Overview
Objective: Accurately classify images of cassava leaves into five categories:

Cassava Bacterial Blight (CBB)

Cassava Brown Streak Disease (CBSD)

Cassava Green Mottle (CGM)

Cassava Mosaic Disease (CMD)

Healthy

Approach: Utilize transfer learning with a ResNet50 architecture to fine-tune the model on the cassava leaf dataset.

Platform: Implemented and trained using Kaggle's TPUs for accelerated computation.

🗂️ Repository Structure
The repository includes the following key files:

source codes.py: Contains the main Python script for data preprocessing, model training, and evaluation.

README.md: Provides an overview and instructions for the project.

🚀 Getting Started
To replicate or build upon this project:

Clone the Repository:

bash
Copy
Edit
git clone https://github.com/AElisha001/Cassava-Leaf-Disease-Classification-Project.git
Set Up Kaggle Environment:

Ensure you have a Kaggle account.

Upload the source codes.py script to a new Kaggle notebook.

Enable TPU support in the notebook settings.

Data Access:

The dataset used is from the Kaggle Cassava Leaf Disease Classification Competition.

In the Kaggle notebook, add the competition dataset to your notebook's data sources.

Run the Script:

Execute the cells in the source codes.py script sequentially to train and evaluate the model.

📊 Model Performance
The model's performance metrics, including accuracy, precision, recall, and F1-score, are evaluated on a validation set. These metrics provide insights into the model's ability to generalize to unseen data.

🧠 Key Features
Transfer Learning: Leverages the ResNet50 model pre-trained on ImageNet, allowing for efficient training on the cassava dataset.

TPU Utilization: Employs TPUs to accelerate the training process, reducing computational time.

Data Augmentation: Incorporates techniques such as rotation, flipping, and scaling to enhance model robustness.

Evaluation Metrics: Uses comprehensive metrics to assess model performance, ensuring reliability.

📚 References
Kaggle Cassava Leaf Disease Classification Competition:
