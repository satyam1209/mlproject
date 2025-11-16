import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifact','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            numerical_feature = ['reading_score', 'writing_score']
            categorical_feature = ['gender','race_ethnicity',
                                   'parental_level_of_education','lunch',
                                   'test_preparation_course']
            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy="median")),
                    ('Scaler',StandardScaler())
                ]
            )
            categorical_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )
            logging.info("Numerical feature standrard scaling done")
            logging.info("categorical feature encoding and standrard scaling done")


            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_feature),
                    ('categorical_pipeline',categorical_pipeline,categorical_feature)
                ]
                
            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")
            logging.info('Obtaining preprocessor object')
            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = 'math_score'
            input_feature_train_df = train_df.drop(columns =[target_column_name],axis=1)
            input_feature_test_df = test_df.drop(columns =[target_column_name],axis=1)
            output_feature_train_df = train_df[target_column_name]
            output_feature_test_df = test_df[target_column_name]
            
            logging.info("Applying transfirmation on input features of test and train dataset")

            input_feature_train_df_transformed = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_df_transformed = preprocessor_obj.transform(input_feature_test_df)
            train_arr = np.c_[
                            input_feature_train_df_transformed, np.array(output_feature_train_df)
                            ]
            test_arr = np.c_[
                            input_feature_test_df_transformed, np.array(output_feature_test_df)
                            ]
            logging.info('saving preprocessing objects')
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)
            