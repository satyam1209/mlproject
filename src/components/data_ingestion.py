import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


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
    a,b = data_ingestion.initiate_data_ingestion()