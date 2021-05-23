# Heart Attack Analysis
Analysis of medical data in order to get a better understanding about heart attack risk.
In short, we'd like to answer 3 questions about heart disease:

1. What are some of the key indicators of coronary artery disease?
2. Do genetic/intrinsic factors play a larger role than environmental/behavioral ones?
3. Is there a simple or low-cost way to evaluate one's own risk of heart attack?

Results and discussion are published on this project's Medium article

# Installation and running
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

# Project components
Run the ipython notebooks in the `code/` directory to follow along the data preparation, modeling and evaluation processes.
The Framingham risk model can be found on the `code/framingham.py` module. Code for preprocessing, model training, evaluation and optimization can be found on `data/modeling.py`
