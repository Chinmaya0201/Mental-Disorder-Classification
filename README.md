# Mental Order Classification

This a classification project done with the help python programming language. This  aims project understands how factors such Sadness, Exhuastion and Sleep disorder and etc affect a person mental state and then classify what type of disease they might be diagnosed with.

Libraries Used - 
- numpy
- pandas
- seaborn
- matplotlib
- scikit-learn
- dill
- streamlit

### Stage 1 EDA and Model training

- We start by analyszing the data we have to work with, checking for missing data, duplicate data and then perfoem exploratory data analysis to gather insights from it. The same can be found in [EDA Notebook](https://github.com/Chinmaya0201/Mental-Disorder-Classification/blob/main/Notebook/EDA.ipynb)

- Then for next part we move to develop the machine learning model which will use existing data to predict the diagnose of the patients. The required preprocessing is done to get data in right format for out ML model to understand starting with The same can be found in [Model Trainer](https://github.com/Chinmaya0201/Mental-Disorder-Classification/blob/main/Notebook/Model_Trainer.ipynb)
1. Proprecessing Missing and Null data
2. Since all the columns where object type had to convert them into numerical format. For columns that had a string in format of (number) of out 10 we extracted that number, for columns with simple yes/no answers we converted them into number using a user defined function and for remaining columns used **Label Encoder**.
3. Scaled the data using **Standard Scaler** before feeding in into the model.
4. Since this was a classification problem I used 4 models Decision Tree, Random Forest, KNN and GaussianNB and after evaluating them on test data found the best model which gave the highest accuracy.

#### Best Model - Random Forest with accuracy of 84%


### Stage 2 Modular Coding

- Then I converted all the steps done in above notebook into **modular programming** and **OOPS**.
It included steps - 
1. Data Ingestion
2. Data Preprocessing
3. Data Transformation
4. Model Trainer 

The code for same can we found in **src** folder. 
I also saved the required object such as the trained model object, preprocessor object, Label Encoder object to use it in next steps.

### Stage 3 Prediction Pipeline 
- Using modular coding and OOPS, developed a prediction pipeline which would predict the disease based on the parameter it receives from user and then return the predict result to user. [Predict Pipeline](https://github.com/Chinmaya0201/Mental-Disorder-Classification/blob/main/src/pipeline/predict_pipeline.py)

### Stage 4 User Application
- Last step was to make a frontend application which could be used by user to interact with project. I used streamlit framework to developed this application. [App Source Code](https://github.com/Chinmaya0201/Mental-Disorder-Classification/blob/main/app.py)

- The app is deployed on Streamlit Community Server




