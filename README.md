#Titanic Survival Prediction
Description
This project aims to predict the survival of passengers aboard the Titanic using logistic regression. The dataset used contains various features such as age, sex, class, and fare, which are utilized to predict whether a passenger survived or not.

Installation
To run this code, ensure you have the required dependencies installed. You can install them using pip:

Copy code
pip install matplotlib numpy pandas seaborn scikit-learn
Save to grepper
Usage
Clone the repository or download the train.csv dataset.
Run the main.py script.
Documentation
Approach
Data Loading and Preprocessing: The Titanic dataset (train.csv) is loaded using pandas. Missing values in the 'Age' column are replaced with the mean age, and missing values in the 'Embarked' column are replaced with the mode. The 'Cabin' column is dropped due to many missing values.

Exploratory Data Analysis (EDA): Various visualizations such as count plots are created using seaborn to explore the distribution and relationships between different features and the target variable ('Survived').

Feature Engineering: Categorical columns ('Sex' and 'Embarked') are encoded to numerical values for model compatibility.

Model Training: The dataset is split into features (X) and the target variable (Y). A logistic regression model is trained using the training data.

Model Evaluation: The accuracy score is calculated for both training and test data to evaluate the model's performance.

Findings
There are 891 entries in the dataset with 11 features.
Logistic regression achieved an accuracy of approximately 80% on both training and test data.
Rationale
Logistic regression is chosen as the model due to its simplicity, interpretability, and effectiveness for binary classification tasks like survival prediction.
Missing values are handled by imputing the mean for 'Age' and the mode for 'Embarked'.
Categorical features are encoded to numerical values to make them compatible with the logistic regression algorithm.
License
This project is licensed under the MIT License.

Contributing
Contributions are welcome. Please open an issue or submit a pull request for any improvements or suggestions.

Contact
For any questions or feedback, feel free to contact vutomi mohube at vutomimohube11@gmail.com.

References
Link to dataset
Scikit-learn documentation: https://scikit-learn.org/stable/
####################################################################################################################################################################################################################################################################
#Breast Cancer Detection Using Logistic Regression
Introduction
Breast cancer is one of the most common types of cancer among women worldwide. Early detection plays a crucial role in improving survival rates and treatment outcomes. In this project, we aim to develop a logistic regression model to predict whether a breast tumor is malignant (cancerous) or benign (non-cancerous) based on various features extracted from digitized images of breast tissue.

Dataset
We utilize the Breast Cancer Wisconsin (Diagnostic) dataset, which is publicly available and widely used for machine learning research. The dataset contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. These features describe characteristics of the cell nuclei present in the image, such as radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension.

Approach
Data Loading and Preprocessing: We load the dataset using Pandas and perform basic preprocessing steps such as dropping unnecessary columns ('id' and 'Unnamed: 32') and encoding the diagnosis labels ('M' for malignant and 'B' for benign) to binary values (1 for malignant and 0 for benign). We also normalize the feature values to ensure that they are on a similar scale.

Splitting the Dataset: We split the preprocessed dataset into training and testing sets, with 85% of the data used for training and 15% for testing. This allows us to evaluate the model's performance on unseen data.

Logistic Regression Model Implementation: We implement logistic regression from scratch using Python and NumPy. The logistic regression algorithm learns the optimal weights and bias by minimizing the cost function through gradient descent. The sigmoid function is used to convert the raw model output into probabilities.

Training the Model: We train the logistic regression model on the training data using the implemented algorithm. During training, we update the model parameters iteratively to minimize the cost and improve accuracy.

Evaluation: We evaluate the trained model's performance on both the training and testing datasets. We calculate the accuracy, which measures the percentage of correctly classified instances, for both datasets.

Comparison with Scikit-Learn Logistic Regression: Finally, we compare the performance of our implemented logistic regression model with that of the logistic regression model provided by the scikit-learn library. This allows us to validate our implementation and assess whether it produces similar results to a well-established library.

Findings
Our logistic regression model achieved a training accuracy of approximately 95.24% and a testing accuracy of around 94.19%.
The scikit-learn logistic regression model achieved a slightly higher testing accuracy of approximately 96.51% and a training accuracy of around 96.69%.
Both models demonstrate high accuracy in distinguishing between malignant and benign breast tumors based on the provided features.
Conclusion
The logistic regression model developed in this project shows promising results in breast cancer detection. It demonstrates high accuracy in predicting tumor malignancy based on features extracted from digitized images of breast tissue. Further optimization and fine-tuning of the model could potentially improve its performance and make it even more useful in clinical settings for early cancer detection.
