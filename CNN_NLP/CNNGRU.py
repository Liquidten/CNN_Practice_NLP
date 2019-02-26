#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 14:26:51 2018

@author: sameepshah

"""

import glob as glob
import pandas as pd
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import numpy as np
#from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

#from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import GlobalMaxPooling1D
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from keras.layers import Conv1D, SpatialDropout1D
from keras import optimizers
from keras.callbacks import TensorBoard
from keras import layers
from sklearn.metrics import roc_auc_score
#from keras.datasets import imdb


positive_Test = glob.glob("/home/sshah33/Opioid_Data/Test/Yes/*.txt")
negative_Test = glob.glob("/home/sshah33/Opioid_Data/Test/No/*.txt")
positive_Train = glob.glob("/home/sshah33/Opioid_Data/Train/Yes/*.txt")
negative_Train = glob.glob("/home/sshah33/Opioid_Data/Train/No/*.txt")

def read_files_in_one_dataframe_column(file_name_list):
	result_df_list = []
	for file_name in file_name_list:
		result_df = pd.read_csv(file_name, names=["Cuis"])
		result_df_list.append(result_df)

	sum_result_df = pd.concat(result_df_list)
	
	return 	sum_result_df


df_Test_P = read_files_in_one_dataframe_column(positive_Test)
#print(df_Test_P)
df_Test_N = read_files_in_one_dataframe_column(negative_Test)
df_Train_P = read_files_in_one_dataframe_column(positive_Train)
df_Train_N = read_files_in_one_dataframe_column(negative_Train)


positive_Test_D = pd.DataFrame(df_Test_P)
positive_Test_D["label"] = 0

negative_Test_D = pd.DataFrame(df_Test_N)
negative_Test_D["label"] = 1

Test = pd.concat(objs = [positive_Test_D, negative_Test_D],
					axis = 0,
					join = "outer")

positive_Train_D = pd.DataFrame(df_Train_P)
positive_Train_D["label"] = 0

negative_Train_D = pd.DataFrame(df_Train_N)
negative_Train_D["label"] = 1


Train = pd.concat(objs = [positive_Train_D, negative_Train_D],
					axis = 0,
					join = "outer")
#print(Test)

Test.columns = ['Cuis','labels']
Train.columns = ['Cuis','labels']
Test_file = (Test['Cuis'])
Test_label = Test['labels']
Train_file = (Train['Cuis'])
Train_label = Train['labels']



def textprocessing(Train_file, Train_label,Test_file, Test_label, MAXLEN):
    X_file, X_dev, y_label, y_dev = train_test_split(Train_file, Train_label, test_size = 0.15, random_state=0)
    tokenizer = Tokenizer(lower = False, oov_token='UNK')
    tokenizer.fit_on_texts(X_file)
    vocab_size = len(tokenizer.word_index) + 1
    #print(encoded)# determine the vocabulary size
    print('Vocabulary Size: %d' % vocab_size)
    encoded_train = tokenizer.texts_to_sequences(X_file)
    X_train = pad_sequences(encoded_train, maxlen = MAXLEN, padding='post')
    
    encoded_dev = tokenizer.texts_to_sequences(X_dev)
    X_dev = pad_sequences(encoded_dev, maxlen = MAXLEN, padding='post')
    print(X_train.shape)
    
    encoded_test = tokenizer.texts_to_sequences(Test_file)
    X_test = pad_sequences(encoded_test, maxlen = MAXLEN, padding='post')
    print(X_test.shape)
    
    return X_train, X_dev, X_test, y_label, y_dev, Test_label, vocab_size
    

def maxlength(Train_file):
    CUIsLength = []
    for x in Train_file:
        y = x.split()
        z = len(y)
        #print(z)
        CUIsLength.append(z)
    highest = max(CUIsLength)
    #print(highest)
    return highest

    

if __name__=="__main__":
    
        
    Maxlen = maxlength(Train_file)
    #print("vocabSize: " + str(vocabSize))
    print("Maxlen:" + str(Maxlen))
    
    
    
    X_train, X_dev, X_test, y_label, y_dev, Test_label, vocab_size = textprocessing(Train_file, Train_label,Test_file, Test_label, Maxlen) 
    print(X_train.shape)
    print(y_label.shape)
    print(X_train)
    print(np.array(y_label))
    e = Embedding(vocab_size,150,input_length=Maxlen)
    
    model = Sequential()
    model.add(e)
    '''
    print('Found %s word vectors.' % len(embeddings_index)) 
    print(embedding_matrix.shape)
    '''
    #output_dir = 'model_output/conv2'
    model.add(SpatialDropout1D(0.5))
    model.add(Conv1D(128, 10, activation = 'relu'))
    model.add(MaxPooling1D(5))
    model.add(Dropout(0.5))
    model.add(Conv1D(64,8, activation = 'relu'))
    #model.add(GlobalMaxPooling1D())
    model.add(layers.GRU(32, dropout=0.5, recurrent_dropout=0.7))
    #model.add(GlobalMaxPooling1D())
    model.add(Dense(1, activation = 'sigmoid', bias_regularizer= 'l2'))
    model.summary()
    '''
    modelcheckpoint = ModelCheckpoint(filepath=output_dir+'/weights.{epoch:02d}.hdf5')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    '''
    #sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    Adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='binary_crossentropy',optimizer=Adam ,metrics=['accuracy'])
    tensorboard = TensorBoard(log_dir="logs/test1")
    x = model.fit(np.array(X_train), np.array(y_label), epochs=15, batch_size = 5, validation_data = (np.array(X_dev), np.array(y_dev)),callbacks=[tensorboard])
    #model.summary()
    prediction = model.predict_classes(np.array(X_test), batch_size = 1)
    prediction2 = model.predict_proba(np.array(X_test), batch_size = 1)
    pct_auc = roc_auc_score(Test_label,prediction) * 100
    print('{:0.2}'.format(pct_auc))
    
    '''
    fpr, tpr, _ = sklearn.metrics.roc_curve(Test_label, prediction2)
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    #print(y_test)
    #print(prediction)
    '''
    