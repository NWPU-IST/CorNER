
import pandas as pd


spark = pd.read_csv('spark/cnn_train_new.csv')
eclipse = pd.read_csv('eclipse/cnn_train_new.csv')

total = pd.concat([spark, eclipse])
total.to_csv('cnn_train_new.csv', index= False)
