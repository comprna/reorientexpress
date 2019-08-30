#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 11:20:48 2019

@author: akanksha srivastava
"""

""" This module is used to build, test and use models that predict the correct orientation of cDNA reads. 
It requires to use either experimental, annotation or cDNA mapped data. It can read both fasta and fastq files.
Also reads compressed files in gz format. 
You can either use this module as a standalone application or import it as a module.
To use it, just use the corresponding flag "-train", "-test" or "-predict" on the command line.
Then, provide all the necessary parameters and files.
Type -h for a detailed list of the parameters. 
"""

import  gzip, argparse
import itertools
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt

from collections import OrderedDict
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

## adding the below lines because some time get cudnn error

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Builds, test and uses models for the orientation of cDNA reads.')
    parser.add_argument('-train', default = False, action = 'store_true', 
        help = 'Set true to train a model.')
    parser.add_argument('-test', default = False, action = 'store_true', 
        help = 'Set true to test a model.')
    parser.add_argument('-predict', default = False, action = 'store_true', 
        help = 'Set true to use a model to make predictions')
    parser.add_argument('-data','--d', action = 'store', type = str, required = True,default = False,
        help = 'The path to the input data. Must be either fasta or fastq. Can be compressed in gz.')
    parser.add_argument('-source', '--s', action = 'store', type = str, required = True, choices = ['annotation','experimental','mapped'],
        help = 'The source of the data. Must be either \'experimental\', \' annotation\' or \'mapped\'. Choose experimental for experiments like RNA-direct, annotation for transcriptomes and mapped for mapped cDNA reads. Mapped reads require a paf file to know the orientation.')
    parser.add_argument('-format', '--f', action = 'store', type = str, choices = ['fasta', 'fastq', 'auto'], default = 'auto',
        help = 'The format of the input data. Auto by deafult. Change only if inconsistencies in the name.')
    parser.add_argument('-annotation', '--a', action = 'store', type = str, default = False,
		help = 'Path to the paf file if a mapped training set which requires a paf reference is being used.')
    parser.add_argument('-use_all_annotation', '-aa', action = 'store_true', default = False,
        help = 'Uses all the reads, instead of only keeping antisense,lincRNA,processed_transcript, protein_coding, and retained_intron. Use it also if the fasta has unconventional format and gives errors.')
    parser.add_argument('-reads', '--r', action = 'store', type = int, default = 10e10,
		help = 'Number of reads to read from the dataset.')
    parser.add_argument('-trimming', '--t', action = 'store', type = int, default = False,
		help = 'Number of nucleotides to trimm at each side. 0 by default.')
    parser.add_argument('-verbose', '--v', action = 'store_true', default = False,
		help = 'Whether to print detailed information about the training process.')
    parser.add_argument('-epochs', '--e', action = 'store', default = 1000, type = int,
		help = 'Number of epochs to train the model.')
    parser.add_argument('-output', '--o', action = 'store', default = 'output',
		help = 'Where to store the outputs. using "--train" outputs a model, while using "-predict" outputs a csv. Corresponding extensions will be added.')
    parser.add_argument('-model', '--m', action = 'store',
		help = 'The model to test or to predict with.')
    parser.add_argument('-reverse_all', '--rm', action = 'store', default = False,
		help = 'All the sequences will be reversed, instead of half of them')
    parser.add_argument('-num_sample_batch', '--n', action = 'store', default = 2000, type = int,
        help = 'Number of samples per batch to train the model.')
    parser.add_argument('-win_size', '--w', action = 'store', default = 500, type = int,
        help = 'Window size for spliting the sequence.')
    parser.add_argument('-step_size', '--step', action = 'store', default = 250, type = int,
        help = 'overlapping size on the the sliding window.')
    options = parser.parse_args()


# Helper functions ------

def reverse_complement(dna):
	"""Takes a RNA or DNA sequence string and returns the reverse complement"""
	complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'U':'A', 'N':'N'}
	return ''.join([complement[base] for base in dna[::-1]])


def sliding_window(sequence, window_size, step_size):
    
    """Converts a sequence into sliding windows. Returns a list of sequences.
	- sequence: a string containing only nucleotides.
	- window_size: Windows of equal size sequence. Default is 500 bp.   
	- step_size: overlap required for sliding windows. Default is 250 bp.
	"""
    seq=[]
    
    for start in range(0, len(sequence), step_size):
         end = start + window_size
         
         if end > len(sequence) :
                
                seq.append(sequence[start:len(sequence)]+"N"*(window_size-(len(sequence)-start)))
               
                break
#        
         else:
             seq.append(sequence[start:end])
          
    return seq


def generate_sets(data, labels,  no_test = False):
    """
	Generate sets for the training, validating and testing. The return depends on the parameters.
	- data: train data. A matrix with columns being normalized counter kmers ordered alphabetically and rows as reads.
	
	- labels: an array of 0 and 1 for each row in data. 1 means reverse and 0 means forward.
	
	- no_test: True if the data provided is not going to be used as test, only as training and validation. Increases the model
				performance.
    """
    print('generating sets')

    if no_test:
        X_train, X_cvt, y_train, y_cvt = train_test_split(data, labels, train_size = 0.75, random_state = 0)
        X_train.reset_index(drop=True,inplace=True)
        X_cvt.reset_index(drop=True,inplace=True)
        y_train.reset_index(drop=True,inplace=True)
        y_cvt.reset_index(drop=True,inplace=True)
       
        return X_train, y_train, X_cvt, y_cvt
    else:
        X_train, X_cvt, y_train, y_cvt = train_test_split(data, labels, train_size = 0.75, random_state = 0)
        X_CV, X_test, y_CV, y_test = train_test_split(X_cvt, y_cvt, train_size = 0.50, random_state = 0)
        X_train.reset_index(drop=True,inplace=True)
        X_cvt.reset_index(drop=True,inplace=True)
        y_train.reset_index(drop=True,inplace=True)
        y_cvt.reset_index(drop=True,inplace=True)
        X_test.reset_index(drop=True,inplace=True)
        y_test.reset_index(drop=True,inplace=True)
        return X_train, y_train, X_CV, y_CV, X_test,y_test

def prepare_data(sequences, order = 'forwarded', drop_duplicates = False, 
	paf_path = False, reverse_all = False):
    """
	Prepares a pandas Series containing nucleotide sequences into a pandas dataframe. Returns a pandas
	data frame with the reads and a pandas Series with the labels (0 for
	forward and 1 for reverse).
	- drop_duplicates: drops sequences that are very similar at the end or beggining. 
	- order: can take different values and process data accordingly.
		* order = forwarded: It can process any kind of format if all the sequences are in forward orientation. For example,
							  for RNA-direct or for the transcriptome.
		* order = mixed: Doesn't assume everything is forward. Expects a paf file that must be provided by
					paf_path argument to know the orientation.
		* order = unknown: assumes the order is unknown. Used to predict.  
	- reverse all: All sequence will be used as the forward orientaion and their reverse complement as reverse for training. Otherwise half will be
    used as forward and the other half will be reverse complemented.  
    
    """
    print('Preparing the data')
    if drop_duplicates: 
        sequences = sequences[~sequences.str[:30].duplicated()]
        sequences = sequences[~sequences.str[-30:].duplicated()]
    sequences = sequences[~sequences.str.contains('N')]
    if order == 'forwarded':
        print('Assuming the data provided is all in forward')
        if reverse_all:
            sequences_reverse = sequences.apply(reverse_complement)	
        else:
            sequences_reverse = sequences.sample(sequences.shape[0]//2)
            sequences = sequences.drop(sequences_reverse.index)
            sequences_reverse = sequences_reverse.apply(reverse_complement)
		
        sequences = pd.DataFrame(sequences)
        sequences_reverse = pd.DataFrame(sequences_reverse)
        sequences['s'] = 0
        sequences_reverse['s'] = 1
        sequences = pd.concat([sequences, sequences_reverse])
        labels = sequences['s']
        data = sequences.drop('s', axis = 1)
        data = data.fillna(0)	
        data.reset_index(inplace=True)
        data.columns=["ID","seq"]		
    elif order == 'mixed':
        print('Using a paf file to infer the orientation of reads')
        ids = pd.read_table(paf_path, usecols = [0,4], index_col = 0, header = None)
        ids = ids[~ids.index.duplicated()]
        ids.index.name = 'ID'
        ids.columns = ['strand']
        sequences = pd.DataFrame(sequences, columns = ['seq'])
        sequences['strand'] = ids['strand']
        sequences['strand'] = sequences['strand'].replace(['+','-'], [0,1])
        sequences = sequences.dropna()
        labels = sequences['strand']
        data = sequences.drop('strand', axis = 1)
        data = data.fillna(0)	
        data.reset_index(inplace=True)
        data.columns=["ID","seq"]
       
    elif order == 'unknown':
        labels = sequences
        data = pd.DataFrame(sequences)
        data.reset_index(inplace=True, drop=True)
        data.columns=["seq"]
    else:
        raise NameError('Invalid source format')
	
    print('Data processed successfully')
    return data, labels


# Reading functions ------

def read_experimental_data(path, format_file = 'auto' ,trimming = False, gzip_encoded = 'auto', 
	n_reads = 50000):
	"""Takes a fasta or fastq file and reads it. The fasta can be compressed in gzip format.
	- path: the fasta file path.
	- trimming: allows to trimming while reading the file, so it's faster than doing it afterwards. False for no trimming. 
				Use an integer to trim the sequence both sides for that length.
	- format: can be 'fastq' or 'auto' to autodetect it.
	- gzip_encoded: if True it reads a gzip compressed fasta file. Use False if the fasta is in plain text. 'auto' tries to infer it from the filename.
	- n_reads: number of reads to reads. Tune according to available memory. 
	"""
	sequences = []
	if gzip_encoded == 'auto':
		if path[-2:] == 'gz':
			gzip_encoded = True
		else:
			gzip_encoded = False
	if gzip_encoded:
		file = gzip.open(path, 'rb')
	else:
		file = open(path, 'r')
	if format_file == 'auto':
		if gzip_encoded:
			marker = file.readline().decode()[0]
		else:
			marker = file.readline()[0]
		if marker == '@':
			format_file = 'fastq'
		elif marker == '>':
			format_file = 'fasta'
			raise NameError('Incorrect format: should be fastq for experimental source')
	if format_file == 'fastq':
		n = 4
#	elif format_file == 'fasta':
#		n = 2
	print('Detected file format: ', format_file)
	i = -1
	kept = 0
	for line in file:
		if gzip_encoded:
			line = line.decode()
		line = line.strip()
		i += 1
		if i%n == 0:
			if kept >= n_reads:
				break
			if line.startswith('>'):
				continue
			else:
				kept += 1
				if trimming:
					sequences.append(line.replace('U', 'T')[trimming:-trimming])
				else:
					sequences.append(line.replace('U', 'T'))
	sequences = pd.Series(sequences)
	return sequences

def read_annotation_data(path, format_file = 'auto', n_reads = 50000, trimming = False, gzip_encoded = 'auto', use_all_annotation = False):
	"""
	This function reads data that doesn't come from an experiment but rather from the reference transcriptome.
	- path: path to the transcriptome file in fasta format.
	- n_reads: number of aproximate reads to process.
	- trimming: allows to trimming while reading the file, so it's faster than doing it afterwards. False for no trimming. 
				Use an integer to trim the sequence both sides for that length.
	- gzip_encoded: if True it reads a gzip compressed fasta file. Use False if the fasta is in plain text. 'auto' tries to infer it from the filename.
	"""
	sequences = []
	if gzip_encoded == 'auto':
		if path[-2:] == 'gz':
			gzip_encoded = True
		else:
			gzip_encoded = False
	if gzip_encoded:
		file = gzip.open(path, 'rb')
	else:
		file = open(path, 'r')
	if format_file == 'auto':
		if gzip_encoded:
			line = file.readline()
			marker = line.decode()[0]
		else:
			line = file.readline()
			marker = line[0]
#		if marker == '@':
#			format_file = 'fastq'
		if marker == '>':
			try:
				_ = line.split('|')[-2]
			except:
				print(line)
				print('The file has not the correct format. All the sequence will be kept to avoid errors.')
				use_all_annotation = True
			format_file = 'fasta'
		else:
			raise NameError('Incorrect format: should be fasta for annotation source')
#	if format_file == 'fastq':
#		n = 4
#	elif format_file == 'fasta':
#		n = 2
	kept = 0
	keep_next = False
	for line in file:
		if kept >= n_reads:
			break
		if gzip_encoded:
			line = line.decode()
		if line.startswith('>'):
			if not use_all_annotation:
				sline = line.split('|')
				read_type = sline[-2]
			if use_all_annotation or read_type in ['antisense','lincRNA','processed_transcript', 'protein_coding', 'retained_intron']:
				if keep_next:
					kept += 1
					if trimming:
						sequences.append(sequence[trimming:-trimming])
					else:
						sequences.append(sequence)
				keep_next = True
				sequence = ''
			else:
				keep_next = False
		else:
			if keep_next:
				sequence += line.strip()
	sequences = pd.Series(sequences)
	return sequences

def read_mapped_data(path, n_reads = 50000, trimming = False, gzip_encoded = 'auto', format_file = 'auto'):
	"""
	Reads RNA that has been generated by mapping reads into a reference. We only consider antisense, lincRNA, 
	processed transcripts, protein coding transcripts and retained introns.
	- path: path to the transcriptome file in fastq format.
	- n_reads: number of aproximate reads to process.
	- trimming: allows to trimming while reading the file, so it's faster than doing it afterwards. False for no trimming. 
				Use an integer to trim the sequence both sides for that length.
	- gzip_encoded: if True it reads a gzip compressed fasta file. Use False if the fasta is in plain text. 'auto' tries to infer it from the filename.
	"""
	sequences = {}
	if gzip_encoded == 'auto':
		if path[-2:] == 'gz':
			gzip_encoded = True
		else:
			gzip_encoded = False
	if gzip_encoded:
		file = gzip.open(path, 'rb')
	else:
		file = open(path, 'r')
	if format_file == 'auto':
		if gzip_encoded:
			line = file.readline()
			marker = line.decode()[0]
		else:
			line = file.readline()
			marker = line[0]
	if marker == '@':
			format_file = 'fastq'
	elif marker == '>':
			format_file = 'fasta'
			raise NameError('Incorrect format: should be fastq for mapped source')
	i = 0
	kept = 0
	for line in file:
		if gzip_encoded:
			line = line.decode()
		line = line.strip()
		i += 1
		if i%4 == 0:
			if kept >= n_reads:
				break
			else:
				kept += 1
				if trimming:
					sequences[indentifier] = line[trimming: -trimming]
				else:
					sequences[indentifier] = line
		elif line.startswith('@'):
			indentifier = line.split('\t')[0].split(' ')[0][1:]
	file.close()
	return pd.Series(sequences).replace('U', 'T')

# define function for CNN models ------
    
def char_to_int(seq):
    CHAR_TO_INT = OrderedDict([('N', 0),('A', 1), ('T', 2), ('G', 3), ('C', 4)])
    return [CHAR_TO_INT[x] for x in seq.upper()]

def char_to_int1(seq):
   
    return list(map(char_to_int,seq))

"""This  function takes the index(a) and no. of columns (5, in case of ACGTN). Index is the converted int values of ACGTN."""  
def get_one_hot(a, ncols):
    a=np.array(a)
    out = np.zeros( (a.size,ncols), dtype=np.int32)
    out[np.arange(a.size),a.ravel()] = 1
    out.shape = a.shape + (ncols,)
    return out
def get_one_hot1(s, ncols):
   
    return [get_one_hot(x,5) for x in s]
def shuffle_in_place(a, b):
    np.random.seed(None)
    rng_state = np.random.get_state()
    np.random.set_state(rng_state)
    a=a.iloc[np.random.permutation(len(a))]
    a.reset_index(inplace=True, drop=True)
    np.random.set_state(rng_state)
    b=b.iloc[np.random.permutation(len(b))]
    b.reset_index(inplace=True, drop=True)
    
    return(a,b)

def generator(n, b):
    
    while True:
        
        for i in range(0,n,b):
            yield np.asarray([i, i + b])

def create_idx_generators(pos_class_y_train, neg_class_y_train, num_sample):

    list_idx_generators = []

    list_idx_generators.append(generator(pos_class_y_train.shape[0], int(num_sample/2)))
    list_idx_generators.append(generator(neg_class_y_train.shape[0], int(num_sample/2)))
    return list_idx_generators

def training_batch_generator(pos_class_x_train,neg_class_x_train,pos_class_y_train, neg_class_y_train,list_idx_generators, ncols=5):

    x = []    
       
    y = []
    
    for i in range(len(list_idx_generators)):
        
        idx = next(list_idx_generators[i])
        #print(idx)
        if i==0:
            seq=pos_class_x_train[0][idx[0]:idx[1]].apply(char_to_int)
            seq=seq.apply(get_one_hot,args=(ncols,))
            
            seq = np.array(seq.values.tolist())
            y.append(pos_class_y_train[idx[0]:idx[1]])
        
        elif i==1 :
            seq=neg_class_x_train[0][idx[0]:idx[1]].apply(char_to_int)
            seq=seq.apply(get_one_hot,args=(ncols,))
            
            seq = np.array(seq.values.tolist())
            y.append(neg_class_y_train[idx[0]:idx[1]])
        x.append(seq)
        
       
    
    x = np.vstack(x)
    
    y = np.vstack(y)
    

    return x, y
#
def make_net(input_shape, num_classes):

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(11, 5),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(4, 1)))
    model.add(Conv2D(64, (3, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Conv2D(96, (3, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.4))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam())
    
        
    return model

def train_cnn(x_train,y_train,x_val,y_val,verbose,model_savename,num_sample_batch, epochs):
    ncols=5
    num_classes=2
    # To select equal no of samples from both class make both class equal
    pos_class_y_train=y_train[y_train[0]==0]
    neg_class_y_train=y_train[y_train[0]==1]
    pos_class_x_train=x_train[y_train[0]==0]
    neg_class_x_train=x_train[y_train[0]==1]
    pos_class_y_train.reset_index(inplace=True, drop=True)
    pos_class_y_train.reset_index(inplace=True, drop=True)
    neg_class_y_train.reset_index(inplace=True, drop=True)
    neg_class_x_train.reset_index(inplace=True, drop=True)
    pos_class_x_train.reset_index(inplace=True, drop=True)
    
    np.random.seed(0)
    if  len(pos_class_x_train)< len(neg_class_x_train):
        ind=np.random.randint(len(pos_class_x_train), size=(1, len(neg_class_x_train)-len(pos_class_x_train)))
        pos_class_x_train=pos_class_x_train.append(pos_class_x_train.iloc[ind[0],:])
        pos_class_y_train=pos_class_y_train.append(pos_class_y_train.iloc[ind[0],:])
        pos_class_x_train.reset_index(inplace=True,drop=True)
        pos_class_y_train.reset_index(inplace=True,drop=True)
    elif  len(neg_class_x_train)<len(pos_class_x_train) :
        ind=np.random.randint(len(neg_class_x_train), size=(1, len(pos_class_x_train)-len(neg_class_x_train)))
        neg_class_x_train=neg_class_x_train.append(neg_class_x_train.iloc[ind[0],:])
        neg_class_x_train.reset_index(inplace=True,drop=True)
        neg_class_y_train=neg_class_y_train.append(neg_class_y_train.iloc[ind[0],:])
        neg_class_y_train.reset_index(inplace=True,drop=True)


    num_batches_per_epoch=round(len(y_train)/num_sample_batch)
    
    ## format x_val and y_val
        
    seq=x_val[0].apply(char_to_int)
    x_val=seq.apply(get_one_hot,args=(ncols,))
    
    x_val = np.array(x_val.values.tolist())
    y_val = keras.utils.to_categorical(y_val, num_classes)
    
    ## determine input shape depending on channel first or last 
    
    rows, cols = x_val.shape[1], ncols ## rows=window size or len of sequence and cols= number of one hot encoding 
    if K.image_data_format() == 'channels_first':
       input_shape = (1, rows, cols)
    else:
        input_shape = (rows, cols, 1)
        
    net = make_net(input_shape, num_classes)
    
    ## reformat x_val with channel info
    x_val = x_val.reshape(x_val.shape[0],input_shape[0], input_shape[1], input_shape[2])
    
    
    print('Training Started...')
    best_valid_loss = 1E6
    best_valid_loss_epoch = 0
    early_stoppage_patience = 20
    
    
    use_cyclic_LR = True
    if use_cyclic_LR:
        
        #FOR ADAM ONLY
        #https://github.com/bckenstler/CLR
        max_lr = 0.001
        base_lr = 0.00001
        step_size = float(num_batches_per_epoch)
        gamma = 0.99994
        cycle_iterations = 0
        K.set_value(net.optimizer.lr, base_lr)
    list_lr = []
        
    for epoch in range(epochs):
        
        ## shuffle in place for better generalization
        pos_class_x_train,pos_class_y_train=shuffle_in_place(pos_class_x_train,pos_class_y_train)
        neg_class_x_train,neg_class_y_train=shuffle_in_place(neg_class_x_train,neg_class_y_train)
        
        list_idx_generators = create_idx_generators(pos_class_y_train, neg_class_y_train,num_sample_batch)
        
        x_tr = []
            
        y_tr = []
        
        #shuffle data here! 
        
        for bnum in range(num_batches_per_epoch):
            
            x_tr, y_tr = training_batch_generator(pos_class_x_train,neg_class_x_train,pos_class_y_train, neg_class_y_train,list_idx_generators)
            x_tr = x_tr.reshape(x_tr.shape[0],input_shape[0], input_shape[1], input_shape[2])
            y_tr = keras.utils.to_categorical(y_tr, num_classes)
            net.train_on_batch(x_tr, y_tr)
            if use_cyclic_LR:
    
                list_lr.append(K.get_value(net.optimizer.lr))
                cycle_iterations = cycle_iterations + 1
                cycle = np.floor( 1 + cycle_iterations/(2.0 * step_size) )
                c = np.abs( cycle_iterations/step_size - 2.0 * cycle + 1 )
                lr = base_lr + (max_lr-base_lr)*np.maximum(0, (1-c))*gamma**(cycle_iterations)
                K.set_value(net.optimizer.lr, lr)
    
            
            
        #validation
        
        train_loss_on_last_batch = net.evaluate(x_tr, y_tr, batch_size=256, verbose=verbose)
        valid_loss = net.evaluate(x_val, y_val, batch_size=256, verbose=verbose)
    
      
    
        #save model is valid loss decreases or last epoch is reached
        if valid_loss < best_valid_loss or epoch == epochs-1 :
            
            print('Saving Best Model')
            best_valid_loss = valid_loss
            best_valid_loss_epoch = epoch
            net.save(model_savename+'_CNN.model')
    
        if (epoch - best_valid_loss_epoch) > early_stoppage_patience:
            print('Training Stopped at epoch %d, best valid_loss = %.5f at epoch %d' % (epoch, best_valid_loss, best_valid_loss_epoch))
            break

def call_train_cnn(kind_of_data, path_data, n_reads, path_paf, trimming,model_savepath,num_sample_batch,epochs,window_size, step_size, verbose = 1, use_all_annotation = False, reverse_all = False):
    """
	Function that automatically reads and processes the data and builds a model with it. Returns the trained model
	and the generated dataset and labelset.
	- kind_of_data: the kind of data used to train the model. Can be:
		* 'experimental' if it comes from RNA direct or similars.
		* 'annotation' if it is the transcriptome reference.
		* 'mapped' if its a mapped cDNA dataset. It requires a paf file to be provided.
	- path_data: path to the data that is going to train the model.
	- n_reads: number of approximate reads to process from the train data. 
	- path_paf: path to the paf file if we are using mapped data.
	- trimming: allows to trimming while reading the file, so it's faster than doing it afterwards. False for no trimming. 
				Use an integer to trim the sequence both sides for that length.
	- ks: maximum lenght of the k-mer counting.
	- full_counting: ensures that all possible lectures windows are used to find the kmers. It makes the process
	slower but more accurate.
	- verbose: can be 0 or 1. 1 means ploting several information related to the training process.
	- epochs: the number of training iterations.
	- checkpointer: if False, the best model is not saved into a file for easy retrieve. If given a name, it saves the model into 
					a file with that name.
	"""
    if kind_of_data == 'experimental':
        sequences = read_experimental_data(path = path_data, trimming = trimming, n_reads = n_reads)
    elif kind_of_data == 'annotation':
        sequences = read_annotation_data(path = path_data, trimming = trimming, n_reads = n_reads, use_all_annotation = use_all_annotation)
    elif kind_of_data == 'mapped':
        sequences = read_mapped_data(path = path_data, trimming = trimming, n_reads = n_reads)
    if path_paf:
        order = 'mixed'
    else:
        order = 'forwarded'
    
    data, labels = prepare_data(sequences, order, False, path_paf, reverse_all = reverse_all)
    labels.reset_index(drop=True,inplace=True)
    data.reset_index(drop=True,inplace=True)
    data=data['seq'].apply(sliding_window, window_size=window_size, step_size=step_size)
    l=data.apply(len)
    
    lab=[]
    for i in range(len(l)):
        lab.append([labels[i]]*l[i])
    lab=list(itertools.chain.from_iterable(lab))
    labels=pd.DataFrame(lab)
    del(lab,l)
    data=list(itertools.chain.from_iterable(data))
    data=pd.DataFrame(data)
    X_train, y_train, X_cvt, y_cvt,_,_= generate_sets(data, labels, no_test = False)
    train_cnn(X_train,y_train,X_cvt,y_cvt,verbose, model_savename=model_savepath,num_sample_batch=num_sample_batch, epochs=epochs)
	
 ## to avoid memory error. To speed the prediction multiprocessing can be applied on the below  function in future

def multipred(x_test_part_df,model,window_size,step_size):
    ncols=5
    rows, cols = window_size, ncols ## rows=window size or len of sequence and cols= number of one hot encoding 
    if K.image_data_format() == 'channels_first':
       input_shape = (1, rows, cols)
    else:
        input_shape = (rows, cols, 1)
    x_test_part_df=x_test_part_df['seq'].apply(sliding_window,window_size=window_size, step_size=step_size)
    x_test_part_df=x_test_part_df.apply(char_to_int1)
    x_test_part_df=x_test_part_df.apply(get_one_hot1,args=(ncols,))
    y_pred_final=[]
    for x_test_m in x_test_part_df:
        y_pred=[]
        for j in x_test_m:
       
            j=j.reshape(1,input_shape[0], input_shape[1], input_shape[2])
            y_pred.append(model.predict(j))
        y_pred_final.append(np.mean(y_pred,axis=0))
    y_pred=list(itertools.chain.from_iterable(y_pred_final))
    #y_pred=pd.DataFrame(y_pred)
    return(y_pred)

def test_model(model, kind_of_data, path_data, n_reads, path_paf, trimming, window_size, step_size, return_predictions = False,reverse_all = False):
    """
	Function that automatically reads and processes the data and test a model with it. Prints several
	metrics about the model performance. !!Use the same parameters as used to train the model!!. 
	- model: trained model. 
	- kind_of_data: the kind of data used to train the model. Can be:
		* 'experimental' if it comes from RNA direct or similars.
		* 'annotation' if it is the transcriptome reference.
		* 'mapped' if its a mapped cDNA dataset. It requires a paf file to be provided.
	- path_data: path to the data that is going to train the model.
	- n_reads: number of approximate reads to process from the train data. 
	- path_paf: path to the paf file if we are using mapped data.
	- trimming: allows to trimming while reading the file, so it's faster than doing it afterwards. False for no trimming. 
				Use an integer to trim the sequence both sides for that length.

	- return_predictions: if True, the predictions and labels are returned with the metrics.
    """
    if kind_of_data == 'experimental':
        sequences = read_experimental_data(path = path_data, trimming = trimming, n_reads = n_reads, format_file =  options.f)
    elif kind_of_data == 'annotation':
        sequences = read_annotation_data(path = path_data, trimming = trimming, n_reads = n_reads, format_file =  options.f)
    elif kind_of_data == 'mapped':
        sequences = read_mapped_data(path = path_data, trimming = trimming, n_reads = n_reads)
    if path_paf:
        order = 'mixed'
    else:
        order = 'forwarded'
    data, labels = prepare_data(sequences, order, False, path_paf, reverse_all = reverse_all)
    y_pred_all=[]
    if data.shape[0]<300:
        chunk_size=1
    else:
        chunk_size = int(data.shape[0] / 300)
    for start in range(0, data.shape[0], chunk_size):
        x_test_part=data.iloc[start:start + chunk_size]
        y_pred_all.append(multipred(x_test_part,model=model,window_size=window_size, step_size=step_size))
    y_pred_all=list(itertools.chain.from_iterable(y_pred_all))
    y_pred_all=pd.DataFrame(y_pred_all)
    y_pred_all.columns=["0","predictions_CNN"]
	
    print('----------------------Test Results-----------------------\n')
    print(classification_report(labels,y_pred_all["predictions_CNN"].round()))
    print('---------------------------------------------------------\n')
    if return_predictions:
        return y_pred_all["predictions_CNN"], labels

def make_predictions(model, kind_of_data, path_data, n_reads, path_paf, trimming, window_size, step_size):
    if kind_of_data == 'experimental':
        sequences = read_experimental_data(path = path_data, trimming = trimming, n_reads = n_reads, format_file = options.f)
    elif kind_of_data == 'annotation':
        sequences = read_annotation_data(path = path_data, trimming = trimming, n_reads = n_reads, format_file = options.f)
    elif kind_of_data == 'mapped':
        sequences = read_mapped_data(path = path_data, trimming = trimming, n_reads = n_reads, format_file = options.f)
    data, labels = prepare_data(sequences, 'unknown',  True, path_paf)
    y_pred_all=[]
    if data.shape[0]<300:
        chunk_size=1
    else:
        chunk_size = int(data.shape[0] / 300)
    for start in range(0, data.shape[0], chunk_size):
        x_test_part=data.iloc[start:start + chunk_size]
        y_pred_all.append(multipred(x_test_part,model=model,window_size=window_size, step_size=step_size))
    y_pred_all=list(itertools.chain.from_iterable(y_pred_all))
    y_pred_all=pd.DataFrame(y_pred_all)
    y_pred_all.columns=["0","predictions_CNN"]
    data['predictions'] = y_pred_all["predictions_CNN"]
    data['orientation'] = 0
    data.loc[data['predictions'] > 0.5, 'orientation'] = 1	
    data.loc[data['predictions'] > 0.5, "seq"] = data["seq"].apply(reverse_complement)
    data.loc[data['predictions'] < 0.5, 'predictions'] = 1 - data['predictions']
    data.columns = ['ForwardSequence', 'Score', 'Orientation']
    data.to_csv(options.o+'.csv')


if __name__ == '__main__':
	if options.train:
		print('\n----Starting Training Pipeline----\n')
       
		call_train_cnn(options.s, options.d, options.r, options.a, options.t, 
			  options.o ,options.n, options.e,options.w,options.step,options.v, options.use_all_annotation, options.rm)

	elif options.test:
		print('\n----Starting Testing Pipeline----\n')
		model = load_model(options.m)
		test_model(model, options.s, options.d, options.r, options.a, options.t,options.w,options.step, True, options.rm)

	elif options.predict:
		print('\n----Starting Prediction Pipeline----\n')
		model = load_model(options.m)
		print('Model successfully loaded')
		print(model.summary())
		make_predictions(model, options.s, options.d, options.r, options.a, options.t,options.w,options.step)
		print('Predictions saved to:', options.o+'.csv')
