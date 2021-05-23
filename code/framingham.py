import numpy as np

class FraminghamModel:
    """
    Model based on the Framingham Score for heart disease risk assessment.
    source = https://en.wikipedia.org/wiki/Framingham_Risk_Score
    """
    def __init__(self, sex_col='sex_M', age_col='age', chol_col='chol', blood_press_col='blood_press_s_rest', smoker_col='smoker'):
        self.age_score = {'F': {34: -7, 39: -3, 44: 0, 49: 3, 54: 6, 59: 8, 64: 10, 69: 12, 74: 14, np.inf: 16},
                           'M': {34: -7, 39: -4, 44: 0, 49: 3, 54: 6, 59: 8, 64: 10, 69: 11, 74: 12, np.inf: 13},
                          }
        self.chol_score = {'F': {39: {160: 0, 199: 4, 239: 8, 279: 11, np.inf: 13},
                                49: {160: 0, 199: 3, 239: 6, 279: 8, np.inf: 10},
                                59: {160: 0, 199: 2, 239: 4, 279: 5, np.inf: 7},
                                69: {160: 0, 199: 1, 239: 2, 279: 3, np.inf: 4},
                                np.inf: {160: 0, 199: 1, 239: 1, 279: 2, np.inf: 2}
                               },
                           'M': {39: {160: 0, 199: 4, 239: 7, 279: 9, np.inf: 11},
                                49: {160: 0, 199: 3, 239: 5, 279: 6, np.inf: 8},
                                59: {160: 0, 199: 2, 239: 3, 279: 4, np.inf: 5},
                                69: {160: 0, 199: 1, 239: 1, 279: 2, np.inf: 3},
                                np.inf: {160: 0, 199: 0, 239: 0, 279: 1, np.inf: 1}
                               },
                          }

        self.smoker_score = {'F': {39: 9, 49: 7, 59: 4, 69: 2, np.inf: 1},
                             'M': {39: 8, 49: 5, 59: 3, 69: 1, np.inf: 1}
                            }
        
        self.hdl_score = {40: 2, 49: 1, 59: 0, np.inf: -1}

        self.bp_score = {'F': {'untreated': {120: 0, 129: 1, 139: 2, 159: 3, np.inf: 4},
                           'treated': {120: 0, 129: 3, 139: 4, 159: 5, np.inf: 6}
                           },
                     'M': {'untreated': {120: 0, 129: 0, 139: 1, 159: 1, np.inf: 2},
                           'treated': {120: 0, 129: 1, 139: 2, 159: 2, np.inf: 3}
                           }
                    }

        self.result_dict = {'F': {9: 0, 12: 0.01, 14: 0.02, 15: 0.03, 16: 0.04, 17: 0.05, 18: 0.06, 19: 0.08, 20: 0.11, 21: 0.14, 22: 0.17, 23: 0.22, 24: 0.27, np.inf: 0.3},
                            'M': {1: 0, 4: 0.01, 6: 0.02, 7: 0.03, 8: 0.04, 9: 0.05, 10: 0.06, 11: 0.08, 12: 0.1, 13: 0.12, 14: 0.16, 15: 0.2, 16: 0.25, np.inf: 0.3}
                            }

        self.sex_col=sex_col
        self.age_col=age_col
        self.chol_col=chol_col
        self.blood_press_col=blood_press_col
        self.smoker_col=smoker_col
    
    def predict(self, X, bp_treatment='untreated'):
        try:
            sex, age, chol, bp = X[self.sex_col], X[self.age_col], X[self.chol_col], X[self.blood_press_col]
            smoker = X[self.smoker_col]
            

            total_score = 0
            
            # Age
            for k, v in self.age_score[sex].items():
                if age <= k:
                    age_score = v
                    break
            total_score += age_score
            
            # Total Cholesterol
            for k, v in self.chol_score[sex].items():
                if age <= k:
                    for c, p in self.chol_score[sex][k].items():
                        if chol <= c:
                            chol_score = p
                            break
                    break
            total_score += chol_score
            
            # Smoking
            if smoker:
                for k, v in self.smoker_score[sex].items():
                    if age <= k:
                        smoker_score = v
                        break
                total_score += smoker_score

            #HDL
            # TODO

            # Blood Pressure
            for k, v in self.bp_score[sex][bp_treatment].items():
                if bp <= k:
                    bp_score = v
                    break
            total_score += bp_score

            # Result
            for k, v in self.result_dict[sex].items():
                if total_score <= k:
                    result = v
                    break

        except:
            result = np.nan

        return result


def test_framingham():
    import pandas as pd

    fm = FraminghamModel()

    df = pd.read_csv('data/processed/clean_and_filled.csv')
    df.smoker = df.smoker.replace({1: 'yes', 0: 'no'})
    df.sex_M = df.sex_M.replace({1: 'M', 0: 'F'})

    return df.apply(fm.predict, axis=1)

if __name__ == '__main__':
    print(test_framingham())



        

