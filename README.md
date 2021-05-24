# Heart Disease Risk Analysis
## Introduction
Analysis of medical data in order to get a better understanding about heart attack risk. This project is part of my Udacity Data Scientist Nanodegree program. In short, we'd like to answer the following questions about heart disease:


1. What are some of the key indicators of coronary artery disease?

2. Do genetic/intrinsic factors play a larger role than environmental/behavioral ones?

3. Is there a simple or low-cost way to evaluate one's own risk of heart attack


General results and discussion are published on this project's [Medium article](https://guischmitd.medium.com/dealing-with-the-most-common-cause-of-death-in-the-world-c8b4fb509ac1)

A technical summary of the analysis in terms of the steps outlined by the CRISP-DM framework can be found on the `CRISP-DM.md` file at the root of this repository. 


## Installation and running
### Copy and paste instructions
For running the code and notebooks, first create and activate a virtual environment (instructions for linux or windows with git bash)
```
# on linux
virtualenv -p python3 .venv
source .venv/bin/activate
```

```
# on windows + git bash
python -m virtualenv .venv
source .venv/Scripts/activate
```

Install python dependencies by running
```
pip install -r requirements.txt
```

### Packages used
- `scikit-learn` for statistical modeling and evaluation
- `pandas`/`numpy` for data manipulation and storing
- `matplotlib`/`seaborn` as my preferred plotting libraries
- `jupyter` for running ipython notebooks
- `tqdm` and `tabulate` for progress bars and enabling pandas output to markdown and other table formats

---
## Project components
Run the ipython notebooks in the `code/` directory to follow along the data preparation, modeling and evaluation processes.
The Framingham risk model can be found on the `code/framingham.py` module. Code for preprocessing, model training, evaluation and optimization can be found on `data/modeling.py`

1. `code\1_process_raw_data.ipynb` contains the code for parsing the original `.data` files as well as performing initial cleanup and variable remapping.
2. `code\2_key_indicators.ipynb` encompasses the **Data Preparation** portion of CRISP-DM, dealing with missing values by dropping or filling columns and creating dummies for categorical variables, as well as an investigation of feature correlations for answering **question 1**.
3. `code\code\3_initial_modeling.ipynb` is concerned with testing the preprocessing, modeling, evaluation and optimization code found in `code/modeling.py` for selecting a model architecture for further analyses.
4. `code\4_genetic_vs_environmental.ipynb` contains the code for exploring **question 2**. The process consists on using subsets of the available features and comparing models for drawing conclusions.
5. `code\5_low-cost_model.ipynb` is concerned with answering **question 3** by using all the prior knowledge and training a model without features that are costly to measure (like specialized medical tests). The resulting model is compared to the full feature model trained on **3.** and to the Framingham risk score model, implemented in `code/framingham.py`.


---


## Results

**1. What are some of the key indicators of coronary artery disease?**
As a whole, the correlation analysis indicates that the key indicators to keep an eye on for predicting heart disease risk are age (older), sex (male), exercise induced angina (chest pain), exercise endurance (longer periods and higher heart rates), and specific medical test results like ST wave depression and Thallium-201 Scintigraphy results.

**2. Do genetic/intrinsic factors play a larger role than environmental/behavioral ones?**
Overall, genetic features prevailed as better predictors of CAD risk, especially considering the recall (or sensitivity) score. The scores were not on par with the ones found earlier when training the models with all available features, which was partly expected. Restricting the models to so few features clearly had a negative impact on their prediction capabilities, but the comparison still holds.

**3. Is there a simple or low-cost way to evaluate one's own risk of heart attack?**
We were able to get really close to the full model without using any of the more costly and complex test results, we were even able to achieve a higher recall, even if at the expense of precision. This tells us that, at least for this sample, we could predict the heart disease risk of a patient to a fairly high degree of confidence without the need of expensive equipment and highly specialized staff.

Further result discussion and details can be found on this project's [Medium article](https://guischmitd.medium.com/dealing-with-the-most-common-cause-of-death-in-the-world-c8b4fb509ac1)