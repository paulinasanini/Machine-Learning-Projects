# Iris Flower Classification
-----------

## Project Description:

The Iris Flower Classification project is a ML task that involves predicting the species of an Iris flower based on its physical attributes. Using a well-known dataset, the project aims to build a classification model that can accurately distinguish between three species of Iris flowers: Iris Setosa, Iris Versicolor, and Iris Virginica. 

![MasterHead](https://www.embedded-robotics.com/wp-content/uploads/2022/01/Iris-Dataset-Classification-1024x367.png)

Image taken from: https://www.embedded-robotics.com/wp-content/uploads/2022/01/Iris-Dataset-Classification-1024x367.png

The dataset, originally introduced by Ronald A. Fisher (1936), consists of 150 samples with four key numerical features (sepal length, sepal width, petal length, and petal width). It is widely available in the  UCI Machine Learning Repository or in Scikit-learn dataset. Due to its balanced nature and simplicity, it is widely used for testing and comparing different machine learning algorithms.

## Problem Statement

Given the Iris dataset, this project aims to implement and evaluate various machine learning classifiers to build a model capable of accurately predicting the species of a given flower based on its physical characteristics.

While Iris Setosa is easily distinguishable due to its well-separated feature distribution, Iris Versicolor and Iris Virginica have overlapping characteristics, making classification a more complex challenge.

## Objectives

The primary goals of this project are to:

1. Explore and preprocess the Iris dataset to understand patterns and relationships between features.
2. Develop and compare different ML models for species classification.
3. Evaluate model performance using different classification metrics to determine the best model for this problem.
4. Visualize results to gain insights into the classification process.
5. Deploy the best model for real-time classification of Iris flowers.

## Key Project Details
- **Dataset Source:** UCI Machine Learning Repository, Scikit-learn
- **Number of Observations:** 150 samples
- **Number of Features:** 4 (Sepal & Petal dimensions in cm)
  **Number of Classes:** 3 (Iris Setosa, Iris Versicolor, Iris Virginica)
- **Type of Problem:** Multiclass Classification
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score
- **Algorithms Considered:** Logistic Regression, K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Decision Trees, Random Forest, Neural Networks
- **Challenges:** Overlapping feature distributions between Versicolor and Virginica

## Conclusion
After extensive data exploration, preprocessing, and evaluation of various Supervised Machine Learning models alongside a Neural Network, using a range of classification metrics such as precision, recall, F1-score, and accuracy, the best model for classifying the Iris flower into its three species (Setosa, Versicolor, and Virginica) was found to be the MLP (Multi-Layer Perceptron). Additionally, the following conclusions can be drawn:

- Data Exploration: Through a thorough examination of the dataset, we gained insights into the characteristics and distributions of features. 
- Data Preprocessing: Essential preprocessing steps were applied, such as handling missing values and encoding categorical variables, to prepare the dataset for the modeling phase.
- Model Selection: After experimenting with multiple machine learning models, the Neural Network (NN) MLP was chosen as the final model due to its ability to effectively capture complex relationships between the features. It provided the best performance in classifying the Iris species compared to other models.
- Model Training and Evaluation: The Neural Network (NN) MLP model was trained on the dataset and evaluated using various metrics. The model demonstrated excellent classification performance, making it the best choice for this task.
- Challenges and Future Work: The future work could involve experimenting with deeper network architectures or different activation functions to enhance the model's performance further.
- Practical Application: The Iris flower classification model can be applied in real-world scenarios like botany and agriculture, where automated identification of Iris species can aid in research, cultivation, and conservation efforts.

In conclusion, the Iris flower classification project successfully utilized the Neural Network (NN) MLP model to classify Iris species. The model's outcomes provide valuable insights into species differentiation and have potential applications in fields like botany and horticulture. With further refinements, this approach could lead to even more accurate and robust classification models in the future.

## References
https://data-flair.training/blogs/iris-flower-classification/
https://raw.githubusercontent.com/Apaulgithub/oibsip_task1/main/Iris.csv
https://github.com/Apaulgithub/oibsip_taskno1/blob/main/Iris_Flower_Classification.ipynb

------
