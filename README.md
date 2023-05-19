# Supervised Machine Learning: Predicting Flight Delays 

---
<br>
<br>
This repository contains all files rquired to reproduce and review the prediction model.
Credits to the following contributors:

- [**Shraddha Patel** ](https://github.com/SHRADDHAYOGIN/)
- [**Patrick Okwir** ](https://github.com/pokwir)

<br>
<br>
### Data
<br>
This model was trained using data colelcted from air travel in the United States between 2018 and 2019, it contains flights, fuel, and passenger data. 

|Dataset   | Description   |
|---|---|
|**flights**   |The departure and arrival information about flights in US in years 2018 and 2019.   |
|**fuel_comsumption**   |The fuel comsumption of different airlines from years 2015-2019 aggregated per month.   |
|**passengers**   |The passenger totals on different routes from years 2015-2019 aggregated per month.   |
| **flights Test**   |The departure and arrival information about flights in US in January 2020.   |

<br>
<br>
The submission CSV (link here) is a prediction of delays for the first 7 days of January 2020 (Jan 1 to 7).
<br>
<br>

For reproducibility, we have included a google drive link to raw data samples that can be used to follow/ reproduce the project. 
<br>

Enjoy!
<br>
<br>

---

## 1. Process
<br>
<br>
<br>
<br>

![Image](Images/Process.png)

- **Problem Definition:** For this project, we investigated flight delays in the United States between 2018 and 2019, we used this data to build and train the model. Each record in the flights dataset describes a flight. The artributes of each feature can be found in the [data description file](https://github.com/lighthouse-labs/mid-term-project-I/blob/master/data_description.md)
<br>

- **Loading the Dataset:** We connected to the AWS postgres database provided to fetch a sizable representative sample (95% confidence level) for all flights. The flight information was used to link to — [fuel consuption](https://github.com/lighthouse-labs/mid-term-project-I/blob/master/data_description.md), and [passengers](https://github.com/lighthouse-labs/mid-term-project-I/blob/master/data_description.md) tables. 
<br>

- **Exploratory Data Analysis:** This focused on structure, answering barinstormed exploratory questions, generating a data profile that would inform feature selection, normalization, and standardization procedures. 
<br>

- **Evaluate Algorithms:** This involved  finding a subset of machine learning algorithms that are good at exploiting the structure of our data and can best answer the hypothesis. This included seperating validation datasets, defining test options using scikit-learn such as cross validation and evaluation metrics to use, spot-checking a suite of linear and non linear models, and comparing estimated accuracy. 
<br>

- **Improving Accuracy:** Once a shorlist was finalized, we employed a search of tuning parameters using scikit-learn that would yield the best results and combined combined predictions of multiple models using ensemble. 
<br>

- **Finalizing the Model:** The best model was chosen. We made preditctions on the [flight test dataset]() and recorded our predictions in [this csv file](). Model was saved for future use. 
  

## 2. Results


## 3. Future Work



