# WIP - Come back later

## Business understanding: Heart disease
Coronary artery disease is the most common cause of death globally. Let that sink in. It makes up 15.6% of all deaths globally, and affected upwards of 110 million people in 2015. In the US, according to the CDC, 1 in every 4 deaths is due to heart diseases, of which CAD is the most prevalent.
CAD is characterized by reduced blood flow to the heart due to narrowing of coronary arteries caused by the build-up of plaque (or lesions) on the walls of these arteries. These lesions or fatty build-ups are called atherosclerosis. The reduced blood flow can cause ischemia to the muscles of the heart which leads to myocardial infarction (a.k.a. heart attacks).
There are some common methods used to diagnose CAD, including electrocardiography (at rest and during exercise), scintigraphy and coronary angiography. These methods require specialized equipment to be performed and a trained physician to evaluate the results.
## Data understanding: Our dataset
*Source, original study/papers, EDA, ambiguous variables, distributions*

The original dataset was compiled by Detrano et al. in a scientific partnership between 5 hospitals in Europe and the US, including the Cleveland Clinic in Ohio, the Hungarian Institute of Cardiology, the Veterans Administration Medical Center in California and the University hospitals of Zurich and Basel, in Switzerland.
The dataset is comprised of 76 columns and 920 rows. Most research and online sources (e.g. Kaggle) of the data focuses only on the patients from Cleveland (303 rows) and a subset of 14 columns (13 features and 1 target) for heart disease classification. "Heart disease" as a target is defined as at least 1 of 4 possible coronary arteries showing >50% narrowing on the angiogram, an invasive medical test.
I started work investigating this subset of the data available on Kaggle, but questions regarding feature descriptions led me further into the research and eventually I stumbled upon the original source. While researching the column descriptions I consulted with my brother, who is currently in med school, for a better understanding of the variables and common procedures in cardiology. During these conversations, he pointed me towards the Framingham Risk Score, a non-invasive method for assessing long term risk of myocardial infarction (or heart attack) based on patient features such as age, sex, cholesterol levels and habits like smoking. The fact that there was a lot of information about smoking in the original dataset, but none in the Kaggle subset was a clear indication that I could probably mine better information from the full data, and perhaps even compare the results of the Framingham test to my own. We'll look at that again when answering question 3.

## Data preparation: 80% of the work
*Missing samples, irrelevant columns, feature transformations, categoricals*

Although the Cleveland subset of the data is well formatted and clean, the original source is far from that. Considering it was collected in the 80s, I was not expecting any common file formats, and that assumption was right. The data are scattered between 4 .data files with slightly different structures, and the one pertaining to Cleveland had a corrupted portion. I will not go into details about this step, but you can find my parsing code in my repo on github.
## Modeling: Turning numbers into magic
*Correlations, linear and ensemble models, train and test results*

## Evaluation