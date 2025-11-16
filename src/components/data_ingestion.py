import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation,DataTransformationConfig
from src.components.model_trainer import ModelTrainer,ModelTrainingConfig

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifact','train.csv')
    test_data_path:str=os.path.join('artifact','test.csv')
    raw_data_path:str=os.path.join('artifact','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered Data Ingestion method")
        try:
            df = pd.read_csv('notebook\stud.csv')
            logging.info("Reading data completed as dataframe")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path)
            logging.info('Saved Raw data')
            logging.info('Train Test split initiated')
            train_set,test_set = train_test_split(df,random_state=42,test_size=0.2)
            train_set.to_csv(self.ingestion_config.train_data_path)
            logging.info('Train Raw data')
            test_set.to_csv(self.ingestion_config.test_data_path)
            logging.info('Test Raw data')
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
             raise CustomException(e,sys)
    

if __name__=='__main__':
    data_ingestion = DataIngestion()
    train_data_path,test_data_path = data_ingestion.initiate_data_ingestion()
    print(train_data_path,test_data_path)
    data_transformation = DataTransformation()
    train_arr,test_arr,preprocessing_obj_path = data_transformation.initiate_data_transformation(train_data_path,test_data_path)
    model_trainer =ModelTrainer()
    r2_score_best_model = model_trainer.initiate_model_training(train_arr,test_arr,preprocessing_obj_path)
    print(r2_score_best_model)