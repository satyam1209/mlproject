import os
import sys
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model
from dataclasses import dataclass
from sklearn.metrics import r2_score




@dataclass
class ModelTrainingConfig:
    trained_model_file_path = os.path.join('artifact','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()
    
    def initiate_model_training(self,train_arr,test_arr,preprocessor_path):
        try:
            logging.info('spliting train and test array')
            X_train,y_train,X_test,y_test = (train_arr[:,:-1],
                                             train_arr[:,-1],
                                             test_arr[:,:-1],
                                             test_arr[:,-1])
            models = {
                        "LinearRegression": LinearRegression(),
                        "Ridge": Ridge(),
                        "Lasso": Lasso(),
                        "ElasticNet": ElasticNet(),
                        "DecisionTreeRegressor": DecisionTreeRegressor(),
                        "RandomForestRegressor": RandomForestRegressor(),
                        "GradientBoostingRegressor": GradientBoostingRegressor(),
                        "KNeighborsRegressor": KNeighborsRegressor(),
                        "SVR": SVR(),
                        "XGBRegressor": XGBRegressor(),
                        "CatBoostRegressor": CatBoostRegressor(verbose=False)
                    }
            report,trained_models = evaluate_model(X_train,X_test,y_train,y_test,models)
            best_model_name=None
            best_model_score = 0
            for model_name,model_score in report.items():
                if model_score>best_model_score:
                    best_model_score = model_score
                    best_model_name=model_name
            if best_model_score<0.6:
                logging.info('Best model has r2 Score less than 60%')
                # raise CustomException('No Best Model Found','')
            logging.info(f"best_model_name = {best_model_name} with r2 score of {best_model_score}")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = trained_models[best_model_name]
            )
            y_pred_test = trained_models[best_model_name].predict(X_test)
            return r2_score(y_test,y_pred_test)


            
        except Exception as e:
            raise CustomException(e,sys)
            
