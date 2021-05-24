# CRISP-DM
## Introduction: the CRISP-DM framework
The Cross-Industry Standard Process for Data Mining, or CRISP-DM is a process for interpreting data in a structured way and gathering relevant information from a business or research standpoint. 
The process is composed of the following steps:
1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment

The following is a summary of the steps taken in the analysis, specifically following the CRISP-DM framework steps.

## Business Understanding
The process starts with the understanding of our business problem. Coronary Artery Disease (or CAD for short) is the most common cause of death globally. It makes up 15.6% of all deaths globally, and affected upwards of 110 million people in 2015. In the US, according to the CDC, 1 in every 4 deaths is due to diseases of the heart, of which CAD is the most prevalent.

CAD is characterized by reduced blood flow to the heart due to narrowing of coronary arteries caused by the build-up of plaque (or lesions) on the walls of these arteries. These lesions or fatty build-ups are called atherosclerosis. The reduced blood flow can cause ischemia to the muscles of the heart which leads to myocardial infarction (a.k.a. heart attacks).

There are some common methods used to diagnose CAD, including electrocardiography (at rest and during exercise), scintigraphy and coronary angiography. These methods require specialized equipment to be performed and a trained physician to evaluate the results.

In this context, I'd like to answer three domain specific questions with the available data:

1. What are some of the key indicators of coronary artery disease?

2. Do genetic/intrinsic factors play a larger role than environmental/behavioral ones?

3. Is there a simple or low-cost way to evaluate one's own risk of heart attack?

## Data understanding and Data Preparation

These steps are commonly carried out in parallel, as there's a lot of back and forth between understanding the variables available and treating their values. In this case, since there was a lot of ambiguous (or outright missing) information in the original files, this step included lots of domain specific research and consulting with my brother, who is currently in med school.

In the code, the data understanding and preparation were carried out in notebooks 1 and 2 (`code/1_preocess_raw_data.ipynb` and `2_key_indicators.ipynb`). 

In the first notebook, there are functions for parsing the original files and compiling the full dataframe. During this phase, I had to understand the column descriptions in the `code/data/raw/schema.txt` file and verify that the values in each column made sense according to the descriptions.

After that, the categorical variables were remapped in an effort to make the parsed dataframe more readable, and to draw a clear distinction between categorical and numeric variables. Before saving the compied dataframe, a histogram was plotted for each remaining numeric column to check for highly skewed distributions and other abnormalities. The rldv5e feature showed an odd distribution, later identified as different data scales being used between the hospitals. These samples were remapped later for consistency.

In the second notebook, the process of data preparation continues by **dealing with missing values**. Some columns were dropped entirely depending on the amount of missing values. Other columns, like the one which encoded whether a patient was a smoker, was filled by a simple heuristic: we used the `smoker_cigs_per_day` and `smoker_years` columns to infer whether a patient smoked where this value was missing.

After dealing with these cases, the remaining missing values were filled depending on the column's variable type. Categorical columns were filled with the mode, while numeric ones were filled with the mean value. Finally, dummies were created for the categorical columns to make them suitable for modeling in the next step.

## Modeling

There were three models chosen for testing: Linear Regression, Random Forests Classifier and Gradient Boosted Tree Classifier. All three models were trained following good practices such as dividing our data into train and test sets, as well as using a static random seed for reproducibility. This step and the next were carried out in the last three notebooks (`code/3_initial_modelling.ipynb`, `code/4_genetic_vs_environmental.ipynb` and `code/5_low-cost_model.ipynb`). Modeling was used to answer the second and third business questions.


## Evaluation

The model evaluation was done by comparing the F1 scores of the trained models, but also computing and comparing accuracy, precision and recall to get a better picture of the model's strengths and weaknesses.

In `code/3_initial_modelling.ipynb`, models trained with all the available features achieved similar perfomances across all metrics, but the Random Forests Classifier showed a slightly better recall score.

In `code/4_genetic_vs_environmental.ipynb`, the model trained solely on genetic variables showed to be a better predictor of CAD than the one trained purely on environmental ones.

Lastly, while comparing the first model from `code/3_initial_modelling.ipynb` with the one trained in `code/5_low-cost_model.ipynb`, it was shown that it's possible to achieve very similar metric scores by only using features which can be measured without the need of specialized equipment or costly medical tests.

As a bonus, a Framingham Risk Score class was implemented and used for a final comparison between the low-cost trained model.

## Deployment

There was no actual deployment of the trained models in the real world, but the solutions found indicate a possibility for model refinement/optimization in the future aiming at deploying it as a tool for heart disease risk assessment available to physicians, health professionals and the general public.


---

## Results

Finally, returning to the business questions raised at the beginning of this investigation, we found the following results:


**1. What are some of the key indicators of coronary artery disease?**
As a whole, the correlation analysis indicates that the key indicators to keep an eye on for predicting heart disease risk are age (older), sex (male), exercise induced angina (chest pain), exercise endurance (longer periods and higher heart rates), and specific medical test results like ST wave depression and Thallium-201 Scintigraphy results.

**2. Do genetic/intrinsic factors play a larger role than environmental/behavioral ones?**
Overall, genetic features prevailed as better predictors of CAD risk, especially considering the recall (or sensitivity) score. The scores were not on par with the ones found earlier when training the models with all available features, which was partly expected. Restricting the models to so few features clearly had a negative impact on their prediction capabilities, but the comparison still holds.

**3. Is there a simple or low-cost way to evaluate one's own risk of heart attack?**
We were able to get really close to the full model without using any of the more costly and complex test results, we were even able to achieve a higher recall, even if at the expense of precision. This tells us that, at least for this sample, we could predict the heart disease risk of a patient to a fairly high degree of confidence without the need of expensive equipment and highly specialized staff.
