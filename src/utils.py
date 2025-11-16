import os 
import sys
import pandas as np
import pandas as pd
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(X_train,X_test,y_train,y_test,models):
    try:
        report = {}
        model_list ={}
        for model_name ,model in models.items():
            model.fit(X_train,y_train)
            model_list[model_name] = model
            y_pred_train = model.predict(X_train)
            y_pred_test  = model.predict(X_test)
            training_score = r2_score(y_train,y_pred_train)
            testing_score = r2_score(y_test,y_pred_test)
            report[model_name] = testing_score
        return report,model_list
    except Exception as e:
        raise CustomException(e,sys)
        