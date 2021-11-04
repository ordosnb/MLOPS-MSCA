# Modified from 

import argparse
import os

# importing necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
# Importing Required Library
import pandas as pd
import lightgbm as lgb
  
# Similarly LGBMRegressor can also be imported for a regression model.
from lightgbm import LGBMRegressor
import joblib

from azureml.core.run import Run
run = Run.get_context()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--kernel', type=str, default='linear',
                        help='Kernel type to be used in the algorithm')
    parser.add_argument('--penalty', type=float, default=1.0,
                        help='Penalty parameter of the error term')

    args = parser.parse_args()
    run.log('Kernel type', np.str(args.kernel))
    run.log('Penalty', np.float(args.penalty))
    
    #Loading Dataset
    # azureml-core of version 1.0.72 or higher is required
    # azureml-dataprep[pandas] of version 1.1.34 or higher is required
    from azureml.core import Workspace, Dataset

    subscription_id = '78aa59ff-61e3-4698-af96-9aac4ecb1457'
    resource_group = 'ml-workspace-rg02'
    workspace_name = 'ml-workspace-intern'

    workspace = Workspace(subscription_id, resource_group, workspace_name)

    dataset = Dataset.get_by_name(workspace, name='train')
    dataset.to_pandas_dataframe()

    # X -> features, y -> label
    X = dataset.drop(columns =['Saleprices'], axis = 1)
    y = dataset.Saleprices

    # dividing X, y into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # training 
    # Creating an object for model and fitting it on training data set 
    model = LGBMRegressor(model = LGBMRegressor()
    model.fit(x_train, y_train)

    # Predicting the Target variable
    pred = model.fit(x_test)
 
    accuracy = model.score(x_test, y_test)
    print(accuracy)

    # model accuracy for X_test
    accuracy = model.score(x_test, y_test)
    print('Accuracy of SVM classifier on test set: {:.2f}'.format(accuracy))
    run.log('Accuracy', np.float(accuracy))
    

    os.makedirs('outputs', exist_ok=True)
    # files saved in the "outputs" folder are automatically uploaded into run history
    joblib.dump(svm_model_linear, 'outputs/model.joblib')


if __name__ == '__main__':
    main()
