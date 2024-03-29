# Deployment_Of_Cancer_Prediction_System
This repo contains code that demonstrates predicting if the patient has cancer or not. 

Using data science methodology 

🐞Machine learing 🐞built with python 🐞deployed with Streamlit 

### Description:
Breast cancer is the most common cancer amongst women in the world. It accounts for 25% of all cancer cases, and affected over 2.1 Million people in 2015 alone. It starts when cells in the breast begin to grow out of control. These cells usually form tumors that can be seen via X-ray or felt as lumps in the breast area.

But COVID-19 pandemic had rapidly changed delivery of cancer care. For breast cancer, a key difference was increased use of neoadjuvant systemic therapy due to deferral of many breast surgeries during the pandemic. We included algorithms based on tumour biology and extent of disease that guide management decisions during the pandemic. These algorithms emphasize if the patient is diagnosed with cancer or not (benign or malignant) by looking at their X-ray reports where a group of radiologists themselves will be able to classify the medical condition of the patient. 

### Project is divided into 2 segments- 
  * Machine LEarning Segment
  * Deployment Segment 
  
Mahine Learning Segment: The dataset does'nt have any missing values hence it was pure set to move ahead with model. SVM classier algorithm is used to classify the begnin and malignant cancer cells. Have tried two different kernels where 'rbf' has prpved to be a good kernel with less loss function. 
Prediction is done on test data as well as the new data and the ouput seems to be accurate. 

Deployment Segmment: Using a virtual environment. The model is pickled, saved and loaded to another file. Streamlit has been used for the deployment. The model is hosted in a local server 

### Predictive model: Overview

•	Created a tool that predicts if the cells are cancerous or non-cancerous 

•	Refered to a dataset in Kaggel

•	Exploration and computations were done on the dataset to know about it

•	Optimized using SVM classifier model
 
 ### Screenshot of webapp Cancer_Predictive_System
![image](https://user-images.githubusercontent.com/111883941/200258055-4c6e1844-2adb-4651-bacb-49cfd74d18f8.png)


### Deployment Segmment- 
Using a virtual environment. The model is pickled, saved and loaded to another file. Created a flask api to deploy the model on local host and also deployed the model on azure cloud. This ML model is hosted in a local/public server
follow the link for app with flask and azure - https://github.com/Munch2022/CancerPrediction_Flask

