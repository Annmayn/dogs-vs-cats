import os
from pathlib import Path
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import h5py
import cv2

# def get_all_files(path, extension=['jpg','png']):
#     """
#         Iterates over the folder and finds all files with provided extension and returns a pandas dataframe object.
#     """
#     # df = pd.DataFrame()
#     # for ext in extension:
#     #     files = Path(path).rglob('*.'+ext)
#     #     for file_loc in files:
#     #         df = df.append([[file_loc, os.path.split(file_loc)[-1].split('.')[0]]])
#     # df.columns = ["X_train, y_train"]
#     # return df

#     for f in os.listdir(path):
#         print(os.path())

def get_all_files(path):
    """
        Iterates over the folder and finds all files with provided extension and returns a pandas dataframe object.
    """
    t1 = time.time()
    data = []
    for file_loc in os.listdir(path):
        data.append([os.path.join(path,file_loc), os.path.split(file_loc)[-1].split('.')[0]])
    df = pd.DataFrame(data)
    df.columns = ["X_train", "y_train"]
    t2=time.time()
    print("Execution time: ",t2-t1)
    return df

def save_as_csv(df, file_name):
    df.to_csv(file_name)


def split_and_save(csv_path, x, y, train_size=0.8, train_fname="train.h5", test_fname="test.h5"):    
    df = pd.read_csv(csv_path, index_col=0)
    X_train_loc, X_test_loc, y_train, y_test = train_test_split(df['X_train'], df['y_train'], train_size=train_size)
    
    enc = LabelEncoder()
    y_train = enc.fit_transform(y_train)
    y_test = enc.transform(y_test)

    image_arr = []
    for image_loc in X_train_loc:
        img = cv2.imread(image_loc)
        image_arr.append(cv2.resize(img,(24,24)).flatten())
    X_train = np.array(image_arr)
    X_train = X_train.T
    with h5py.File(train_fname,'w') as hf:
        hf.create_dataset('X',data=X_train)
        hf.create_dataset('y',data=y_train.reshape(1,-1))

    image_arr = []
    for image_loc in X_test_loc:
        img = cv2.imread(image_loc)
        image_arr.append(cv2.resize(img,(24,24)).flatten())
    X_test = np.array(image_arr)
    X_test = X_test.T
    with h5py.File(test_fname,'w') as hf:
        hf.create_dataset('X',data=X_test)
        hf.create_dataset('y',data=y_test.reshape(1,-1))
    