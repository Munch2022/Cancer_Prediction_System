# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 14:34:43 2022

@author: Manjula 
"""

import numpy as np
import pickle
import streamlit as st
import sklearn

# loading the trained model; copy the path where u saved downloaded trained model
loaded_model= pickle.load(open('C:/Users\Manjula\Desktop/Deploying_ML_model/trained_model.sav', 'rb')) 


# creating a function for prediction 

def cancer_predcition(input_data):
    input_data_arr= np.asarray(input_data)
    inputdata_reshaped= input_data_arr.reshape(1, -1)               # coz it sontains sample(series of values)

    prediction = loaded_model.predict(inputdata_reshaped)              # here instead of clf2, im giving teh loaded_model as teh model is loaded in this variable
    print(prediction)

    if (prediction[0] == 2):
        return 'Its a benign cancer'
    else:
        return 'Its a malignant cancer, Consult doctor immediately'



# lets come out of the function and build something for the streaming part 
# this part will contain the streamlit library

def main():
    
    # giving a title for the web page 
    st.title('Cancer Prediction Web App')
    
    # now we should create variables so that user can input the data
    # getting the input data from the user 
    
    Clump= st.text_input('Enter the Clump size')
    UnifSize= st.text_input('Enter the Unif size')
    UnifShape= st.text_input('Enter shape of the Unif')
    MargAdh= st.text_input('Enter Margadh')
    SingEpiSize= st.text_input('Enter Single Epi Size')
    BareNuc= st.text_input('Enter the number of Bare Nucleus')
    BlandChrom= st.text_input('Enter number of Bland Chromosomes')
    NormNucl= st.text_input('Enter Normal Nucleus')
    Mit= st.text_input('Enter Mit')
    
    
    # code for Prediction;  diagnosis will be empty as it will later contain the o/p of cancer_prediction 
    diagnosis= ''          
    
    # creating a button for prediction
    if st.button('Cancer Test Result'):
        diagnosis= cancer_predcition([Clump, UnifSize, UnifShape, MargAdh, SingEpiSize, BareNuc, BlandChrom, NormNucl, Mit])
    
    st.success(diagnosis)
    
    
if __name__ == '__main__':
    main()
    
# main() function is a function we created for web page

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
