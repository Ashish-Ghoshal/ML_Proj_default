import os
from mlProject import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from mlProject.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config


    
    ## Note: You can add different data transformation techniques such as Scaler, PCA and all
    #You can perform all kinds of EDA in ML cycle here before passing this data to the model

    # I am only adding train_test_spliting cz this data is already cleaned up
    def scaling(self,train_x,test_x):
        # Standarizing with standard scaler
        scaler_x = MinMaxScaler()
        x_train = scaler_x.fit_transform(train_x)
        x_test= scaler_x.transform(test_x)
        
        return x_train,x_test

    def PCA(self,train_x,test_x):
        pca = PCA(random_state=10)
        pca.fit(train_x)
        # Let's take 10 , which explains 98.5 %
        pc_selected = PCA(n_components=10, random_state=10)
        # This is the new transformed data
        x_train_pc = pc_selected.fit_transform(train_x)
        x_test_pc = pc_selected.transform(test_x)
        
        return x_train_pc,x_test_pc


    def train_test_spliting(self):
        data = pd.read_csv(self.config.data_path)
        
       

        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(data)
        print(type(train))
        
        
        train_x = train.drop([self.config.target_column], axis=1)
        test_x = test.drop([self.config.target_column], axis=1)
        train_y = train[[self.config.target_column]]
        test_y = test[[self.config.target_column]]
        print(type(train_y))
        
        train_x,test_x = self.scaling(train_x,test_x)
        train_x,test_x = self.PCA(train_x,test_x)
        
        train_x = pd.DataFrame(train_x)
        test_x = pd.DataFrame(test_x)
        
        train = train_x
        train[self.config.target_column] = train_y
        
        test = test_x
        test[self.config.target_column] = test_y
        
        print(train)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)
    

        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)
