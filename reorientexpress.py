#!/usr/bin/python
""" This module is used to build, test and use models that predict the correct orientation of cDNA reads. 
It requires to use either experimental, annotation or cDNA mapped data. It can read both fasta and fastq files.
Also reads compressed files in gz format. 
You can either use this module as a standalone application or import it as a module.

To use it, just use the corresponding flag "-train", "-test" or "-predict" on the command line.
Then, provide all the necessary parameters and files.
Type -h for a detailed list of the parameters. 

"""
import os, itertools, re
import pandas, math, gzip, numpy, argparse, time
from keras import optimizers
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
from scipy import stats
from Bio import SeqIO
from Bio.Seq import Seq

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Builds, test and uses models for the orientation of cDNA reads.')
    parser.add_argument('-train', default = False, action = 'store_true', 
    help = 'Set true to train a model.')
    parser.add_argument('-test', default = False, action = 'store_true', 
    help = 'Set true to test a model.')
    parser.add_argument('-predict', default = False, action = 'store_true', 
    help = 'Set true to use a model to make predictions')
    parser.add_argument('-output_fastq', default = False, action = 'store_true', 
    help = 'Set true to get the output in fastq format')
    parser.add_argument('-data','--d', action = 'store', type = str, required = True, default = False,
    help = 'The path to the input data. Must be either fasta or fastq. Can be compressed in gz.')
    parser.add_argument('-source', '--s', action = 'store', type = str, required = True, choices = ['annotation','experimental','mapped', 'csv'],
    help = 'The source of the data. Must be either \'experimental\', \' annotation\' or \'mapped\'. Choose experimental for experiments like RNA-direct, annotation for transcriptomes and mapped for mapped cDNA reads. Mapped reads require a paf file to know the orientation.')
    parser.add_argument('-format', '--f', action = 'store', type = str, choices = ['fasta', 'fastq', 'auto'], default = 'auto',
    help = 'The format of the input data. Auto by deafult. Change only if inconsistencies in the name.')
    parser.add_argument('-annotation', '--a', action = 'store', type = str, default = False,
    help = 'Path to the paf file if a mapped training set which requires a paf reference is being used.')
    parser.add_argument('-use_all_annotation', '-aa', action = 'store_true', default = False,
    help = 'Uses all the reads, instead of only keeping antisense,lincRNA,processed_transcript, protein_coding, and retained_intron. Use it also if the fasta has unconventional format and gives errors.')
    parser.add_argument('-kmers', '--k', action = 'store', type = int, required = False, default = 5,
    help = 'The maximum length of the kmers used for training, testing and using the models.')
    parser.add_argument('-fixedkmer', '--fk', action = 'store_true', required = False, default = False,
    help = 'Only use the last kmer length')
    parser.add_argument('-reads', '--r', action = 'store', type = int, default = 10e10,
    help = 'Number of reads to read from the dataset.')
    parser.add_argument('-trimming', '--t', action = 'store', type = int, default = False,
    help = 'Number of nucleotides to trimm at each side. 0 by default.')
    parser.add_argument('-verbose', '--v', action = 'store_true', default = False,
    help = 'Whether to print detailed information about the training process.')
    parser.add_argument('-epochs', '--e', action = 'store', default = 20, type = int,
    help = 'Number of epochs to train the model.')
    parser.add_argument('-output', '--o', action = 'store', default = 'output',
    help = 'Where to store the outputs. using "--train" outputs a model, while using "-predict" outputs a csv. Corresponding extensions will be added.')
    parser.add_argument('-model', '--m', action = 'store',
    help = 'The model to test or to predict with.')
    parser.add_argument('-reverse_all', '--ra', action = 'store_true', default = False,
    help = 'All the sequences will be reversed, instead of half of them')
    parser.add_argument('-reads_to_model', '--rm', action = 'store', type = int, default = int(10e10),
    help = 'Number of reads to use from the read ones')
    parser.add_argument('-one_hot', '--oh', action = 'store_true', default = False,
    help = 'Use one hot encoding instead of kmer counting')
    options = parser.parse_args()


# Helper functions ------

def reverse_complement(dna):
	"""Takes a RNA or DNA sequence string and returns the reverse complement"""
	complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'U':'A', 'N':'N'}
	return ''.join([complement[base] for base in dna[::-1]])

def sequences_to_kmers(seq, ks, only_last_kmer = False, full_counting = False, one_hot = False):
	"""Converts a sequence to kmers counting. Returns a pandas Series object for easier processing.
	- seq: a string containing only nucleotides.
	- ks: maximum lenght of the k-mer counting.
	- only_last_kmer: calculate only the biggest k-mer, but not the others.
	- full_counting: ensures that all possible lectures windows are used to find the kmers. It makes the process
	slower but more accurate.
	- one_hot: if true, instead of counting kmers, it performs one-hot encoding.
	"""
	kmers = {}
	length = len(seq)
	if only_last_kmer:
		starting = ks
	else:
		starting = 1
	if full_counting:
		windows = ks
	else:
		windows = 1
	for k in range(starting, 1+ks):
		for window in range(min(windows, k)):
			for i in range(len(seq)//k):
				subseq = seq[i*k+window: i*k+k+window]
				if not re.match('^[ACTG]+$', subseq): # Ensures we discard ambigous nucleotide sequences.
					continue
				if len(subseq) < k:
					continue
				if subseq in kmers:
					if one_hot:
						kmers[subseq] = 1
					else:
						kmers[subseq] += 1/(length-k+1)
				else:
					if one_hot:
						kmers[subseq] = 1
					else:
						kmers[subseq] =  1/(length-k+1)

	return pandas.Series(kmers)

def check_valid_nucleotides(seq):
	"""
	Used to clean sequences with non-standard nucleotide symbols.
	"""
	if re.match('^[ACTG]+$', seq):
		return True
	else:
		return False

def generate_sets(data, labels, norm = False, do_not_split = False, no_test = False, mn_reads = int(10e10)):
	"""
	Generate sets for the training, validating and testing. The return depends on the parameters.
	- data: train data. A matrix with columns being normalized counter kmers ordered alphabetically and rows as reads.
	- do_not_split: if you want all the data in the same set, but shuffled. 
	- labels: an array of 0 and 1 for each row in data. 1 means reverse and 0 means forward.
	- norm: if True normalizes the data. As the counting kmers are already normalized it's usually not necessary. If 
			the results are not good enought, set True to normalize across samples, which might help.
	- no_test: True if the data provided is not going to be used as test, only as training and validation. Increases the model
				performance.
	"""
	print('generating sets')
	if norm:
		data = normalize(data)
	if do_not_split:
		data = shuffle(data)
		labels = labels.loc[data.index]
		print('sets generated')
		return data, labels
	X_train, X_cvt, y_train, y_cvt = train_test_split(data[:int(mn_reads)], labels[:int(mn_reads)], train_size = 0.75, random_state = 0)
	X_CV, X_test, y_CV, y_test = train_test_split(X_cvt, y_cvt, train_size = 0.50, random_state = 0)
	print('sets generated')
	if no_test:
		return X_train, y_train, X_cvt, y_cvt
	else:
		return X_train, y_train, X_CV, y_CV, X_test,y_test

def prepare_data(sequences, order = 'forwarded', full_counting = True, ks = 5, drop_duplicates = False, 
	paf_path = False, ensure_all_kmers = False, only_last_kmer = False, reverse_all = False,
	one_hot = False):
	"""
	Prepares a pandas Series containing nucleotide sequences into a pandas dataframe with kmers counting. Returns a pandas
	data frame with the normalized kmer counts as columns and the reads as rows and a pandas Series with the labels (0 for
	forward and 1 for reverse).
	- drop_duplicates: drops sequences that are very similar at the end or beggining. 
	- order: can take different values and process data accordingly.
		* order = forwarded: It can process any kind of format if all the sequences are in forward orientation. For example,
							  for RNA-direct or for the transcriptome.
		* order = mixed: Doesn't assume everything is forward. Expects a paf file that must be provided by
					paf_path argument to know the orientation.
		* order = unknown: assumes the order is unknown. Used to predict.  

	- full_counting: full_counting: ensures that all possible lectures windows are used to find the kmers. It makes the process
	slower but more accurate.
	- ks: maximum lenght of the k-mer counting.
	- ensure_all_kmers: if True, it makes sure all the mers are calculated. Use only for small files if some the prediction fails.
	"""
	print('Preparing the data')
	sequences = sequences[sequences.apply(check_valid_nucleotides)]
	if order == 'forwarded':
		print('Assuming the data provided is all in forward')
		if reverse_all:
			sequences_reverse = sequences.sample(sequences.shape[0])
			sequences_reverse.index = sequences_reverse.index + sequences.shape[0]
		else:
			sequences_reverse = sequences.sample(sequences.shape[0]//2)
			sequences = sequences.drop(sequences_reverse.index)
		sequences_reverse = sequences_reverse.apply(reverse_complement)
		sequences = sequences.apply(sequences_to_kmers, ks = ks, full_counting = full_counting, only_last_kmer=only_last_kmer, one_hot = one_hot)
		sequences_reverse = sequences_reverse.apply(sequences_to_kmers, ks = ks, full_counting = full_counting, only_last_kmer=only_last_kmer, one_hot = one_hot)
		sequences = pandas.DataFrame(sequences)
		sequences_reverse = pandas.DataFrame(sequences_reverse)
		sequences['s'] = 0
		sequences_reverse['s'] = 1
		sequences = pandas.concat([sequences, sequences_reverse])
		sequences = sequences.sample(frac = 1)
		labels = sequences['s']
		data = sequences.drop('s', axis = 1)
		data = data.fillna(0)
	elif order == 'mixed':
		print('Using a paf file to infer the orientation of reads')
		ids = pandas.read_table(paf_path, usecols = [0,4], index_col = 0, header = None)
		ids = ids[~ids.index.duplicated()]
		ids.index.name = 'ID'
		ids.columns = ['strand']
		sequences = pandas.DataFrame(sequences, columns = ['seq'])
		sequences['strand'] = ids['strand']
		sequences['strand'] = sequences['strand'].replace(['+','-'], [0,1])
		sequences = sequences.dropna()
		labels = sequences['strand']
		data = sequences.drop('strand', axis = 1)
		data = sequences['seq'].apply(sequences_to_kmers, ks = ks, full_counting=full_counting, only_last_kmer=only_last_kmer, one_hot=one_hot)
		data = data.fillna(0)
	elif order == 'unknown':
		labels = sequences
		sequences = sequences.apply(sequences_to_kmers, ks = ks, full_counting = full_counting, only_last_kmer=only_last_kmer, one_hot=one_hot)
		sequences = pandas.DataFrame(sequences)
		data = sequences.fillna(0)
	else:
		raise NameError('Invalid source format')
	if ensure_all_kmers:
		bases=['A','T','G','C']
		for k in range(ks):
			kmers = [''.join(p) for p in itertools.product(bases, repeat=k+1)]
			for kmer in kmers:
				if kmer not in data.columns:
					data[kmer] = 0
		data = data.sort_index(axis = 1)
	print('Data processed successfully')
	return data, labels

# Reading functions ------

def read_experimental_data(path, format_file = 'auto' ,trimming = False, gzip_encoded = 'auto', 
	n_reads = 50000):
	"""Takes a fasta or fastq file and reads it. The fasta can be compressed in gzip format.
	- path: the fasta file path.
	- trimming: allows to trimming while reading the file, so it's faster than doing it afterwards. False for no trimming. 
				Use an integer to trim the sequence both sides for that length.
	- format: can be 'fasta', 'fastq' or 'auto' to autodetect it.
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
		else:
			raise NameError('Incorrect format')
	if format_file == 'fastq':
		n = 4
	elif format_file == 'fasta':
		n = 2
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
	sequences = pandas.Series(sequences)
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
		file_c = gzip.open(path, 'rb')
		file = gzip.open(path, 'rb')
	else:
		file_c = open(path, 'r')
		file = open(path, 'r')
	if format_file == 'auto':
		if gzip_encoded:
			line = file_c.readline()
			line = line.decode()
			marker = line[0]
		else:
			line = file_c.readline()
			marker = line[0]
		if marker == '@':
			format_file = 'fastq'
			use_all_annotation = True
		elif marker == '>':
			if len(line.split('|')) > 2:
				separator = '|'
			elif len(line.split(' ')) > 2:
				separator = ' '
			else:
				print('The file has not the correct format. All the sequence will be kept to avoid errors.')
				use_all_annotation = True
			format_file = 'fasta'
		else:
			raise NameError('Incorrect format')
	if format_file == 'fastq':
		n = 4
	elif format_file == 'fasta':
		n = 2
	kept = 0
	keep_next = False
	for line in file:
		if kept >= n_reads:
			break
		if gzip_encoded:
			line = line.decode()
		if line.startswith('>'):
			line = line.strip()
			if not use_all_annotation:
				if separator == '|':
					sline = line.split('|')
					read_type = sline[-2]
				elif separator == ' ':
					sline = line.split(' ')
					if ':' in sline[-1]:
						read_type = sline[-1].split(':')[1]
					else:
						use_all_annotation = True
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
	if trimming:
		sequences.append(sequence[trimming:-trimming])
	else:
		sequences.append(sequence)
	sequences = pandas.Series(sequences)
	return sequences

def read_mapped_data(path, n_reads = 50000, trimming = False, gzip_encoded = 'auto', format_file = 'auto'):
	"""
	Reads RNA that has been generated by mapping reads into a reference. We only consider antisense, lincRNA, 
	processed transcripts, protein coding transcripts and retained introns.
	- path: path to the transcriptome file in fasta format.
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
	file.readline()
	file.readline()
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
	return pandas.Series(sequences).replace('U', 'T')


def read_cluster_data(path,trimming = False, n_reads = 50000):
	"""
	"""
	sequences = []
	ids = []
	kept = 0
	q = False
	file = open(path)
	for line in file:
		line = line.strip()
		if kept >= n_reads:
			break
		if line == '+':
			q = True
			continue
		elif q==True:
			q = False
			continue
		if line.startswith('@'):
			ids.append(line[1:37])
		else:
			kept += 1
			if trimming:
				sequences.append(line.replace('U', 'T')[trimming:-trimming])
			else:
				sequences.append(line.replace('U', 'T'))
	sequences = pandas.Series(sequences, index = ids)
	return sequences

# Model functions ------

def plain_NN(input_shape, output_shape, n_layers = 5, n_nodes = 5, step_activation = 'relu',
	final_activation = 'sigmoid',optimizer = False, kind_of_model = 'classification', 
	halve_each_layer = False,dropout = False, learning_rate = 0.0001):
	"""  
	Creates a simple neural network model and returns the model object.
	-input_shape = integer that represents the number of features used by the model.
	-output_shape = integer that represents the number of features the model tries to predict.
	-n_layers = the number of layers in the model.
	-n_nodes = the number of nodes in each layer.
	-step_activation = activation function at each step, can be any that keras uses.
	-final_activation = activation function at the final step, can be any that keras uses.
	-optimizers = if provided, it uses the optimizers delivered.
	-halve each layer = if true, each layer has half the nodes as the previous one.
	-dropout = use drouput layers.
	-learning_rate = the learning rate for the model to learn. 
	"""
	print('Creating model architecture')
	model = Sequential()
	model.add(Dense(n_nodes, activation='relu', input_dim=input_shape))
	if halve_each_layer:
		halver = 2
	else:
		halver = 1
	if dropout:
		model.add(Dropout(0.3))
	for i in range(n_layers-1):
		n_nodes = n_nodes // halver
		model.add(Dense(n_nodes,activation = step_activation))
		if dropout:
			model.add(Dropout(0.3))
	model.add(Dense(output_shape, activation = final_activation))
	if optimizer:
		optimizer = optimizer
	else:
		optimizer = optimizers.RMSprop(lr = learning_rate)
	if kind_of_model == 'classification':
		model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
	if kind_of_model == 'regression':
		model.compile(optimizer = optimizer, loss = 'mse', metrics = ['mae'])
	print(model.summary())
	return model

def fit_network(model, data, labels, epochs = 10, batch_size = 32, verbose = 1 ,checkpointer = False, no_test = True, mn_reads = int(10e10)):
	"""
	Fits a neural network into a model and returns the history to easily analyze the performance.
	Returns the trained model and the training history, for evaluation purposes.
	checkpointer: if given a name, creates a checkpointer with that name.
	- model: model to train the data with. Must have the same input shape as number of variables in the train set.
	- data: train data. A matrix with columns being normalized counter kmers ordered alphabetically and rows as reads.
	- labels: an array of 0 and 1 for each row in data. 1 means reverse and 0 means forward.
	- epochs: number of iterations to train the model. Recomended from 10 to 100. The more data the less epochs are necessary.
	- verbose: whether to print several information related to the training process.
	- batch_size: number of reads to train the model at once during each epochs. 
	- checkpointer: if given a name, creates a checkpointer with that name.
	- no_test: True if the data provided is not going to be used as test, only as training and validation. Increases the model
				performance.

	"""
	if no_test:
		X_train, y_train, X_CV, y_CV = generate_sets(data, labels, no_test = no_test, mn_reads = mn_reads)
	else:
		X_train, y_train, X_CV, y_CV, X_test,y_test = generate_sets(data, labels, mn_reads = mn_reads)
	if checkpointer:
		print('Using Checkpointer')
		model_file = checkpointer+'.model'
		checkpointer = ModelCheckpoint(filepath= model_file,  
			verbose=verbose, save_best_only=True)
		train_time0 = time.time()
		history = model.fit(X_train.values, y_train.values, batch_size=batch_size, 
			epochs=epochs,validation_data=(X_CV.values, y_CV.values), verbose=verbose,
			callbacks = [checkpointer])
		train_time1 = time.time()
		print('Elapsed time while training: ', train_time1 - train_time0, ' seconds')
		model.load_weights(model_file)
		model.save(model_file)
		print('Best model train accuracy: ', model.evaluate(X_train.values, y_train.values))
		print('Best model validation accuracy: ', model.evaluate(X_CV.values, y_CV.values))
	else:
		history = model.fit(X_train.values, y_train.values, batch_size=batch_size, 
			epochs=epochs,validation_data=(X_CV.values, y_CV.values), verbose=verbose)
		if not no_test:
			print(model.evaluate(X_test.values, y_test.values))
	return model, history

def build_kmer_model(kind_of_data, path_data, n_reads, path_paf, trimming, full_counting, ks, verbose = 1,
	epochs = 10, checkpointer = 'cDNAOrderPrediction', use_all_annotation = False, only_last_kmer = False, reverse_all = False,
	mn_reads = int(10e10), one_hot = False):
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
	data, labels = prepare_data(sequences, order, full_counting, ks, False, path_paf, only_last_kmer=only_last_kmer, reverse_all = reverse_all, one_hot = one_hot)
	model = plain_NN(data.shape[1],1, 5, 500, step_activation = 'relu', final_activation = 'sigmoid', 
		optimizer = False, kind_of_model = 'classification', halve_each_layer = True,dropout = True, 
		learning_rate = 0.00001)
	model, history = fit_network(model, data, labels, epochs = epochs, verbose = verbose, checkpointer = checkpointer, batch_size = 64, mn_reads = mn_reads)
	return model, history ,data, labels

def test_model(model, kind_of_data, path_data, n_reads, path_paf, trimming, full_counting, ks, one_hot, return_predictions = False, mn_reads = int(10e10)):
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
	- ks: maximum lenght of the k-mer counting.
	- full_counting: ensures that all possible lectures windows are used to find the kmers. It makes the process
	slower but more accurate.
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
	data, labels = prepare_data(sequences, order, full_counting, ks, False, path_paf, one_hot=one_hot)
	data, labels = (data[:int(mn_reads)], labels[:int(mn_reads)])
	predictions = model.predict(data.values)
	print('----------------------Test Results-----------------------\n')
	print(classification_report(labels,predictions.round()))
	print('---------------------------------------------------------\n')
	if return_predictions:
		return predictions, labels
    
def outformat(path_data, data):
    
    c=-1
    updated_file=[]
    with open(path_data) as handle:
       
        for record in SeqIO.parse(handle, 'fastq'):
            if (data.Orientation[c+1]==1):
                record=record.reverse_complement(id=True,description=True)
            c=c+1
            updated_file.append(record)
        SeqIO.write(updated_file, options.o+'.fastq', "fastq")
        
def make_predictions(model, kind_of_data, path_data, n_reads, path_paf, trimming, full_counting, ks, one_hot):
    if kind_of_data == 'experimental':
        sequences = read_experimental_data(path = path_data, trimming = trimming, n_reads = n_reads, format_file = options.f)
    elif kind_of_data == 'annotation':
        sequences = read_annotation_data(path = path_data, trimming = trimming, n_reads = n_reads, format_file = options.f)
    elif kind_of_data == 'mapped':
        sequences = read_mapped_data(path = path_data, trimming = trimming, n_reads = n_reads, format_file = options.f)
    elif kind_of_data == 'csv':
        sequences = pandas.read_csv(path_data, index_col = 0, squeeze = True, nrows = n_reads)
    data, labels = prepare_data(sequences, 'unknown', full_counting, ks, False, path_paf, one_hot = one_hot, ensure_all_kmers = True)
    predictions = model.predict(data.values)
    data = pandas.DataFrame(labels)
    data['predictions'] = predictions
    data['orientation'] = 0
    data.loc[data['predictions'] > 0.5, 'orientation'] = 1	
    data.loc[data['predictions'] > 0.5, 0] = data[0].apply(reverse_complement)
    data.loc[data['predictions'] < 0.5, 'predictions'] = 1 - data['predictions']
    data.columns = ['ForwardSequence', 'Score', 'Orientation']
    if (kind_of_data == 'experimental'and options.output_fastq):
        outformat(path_data,data)
    else:
        data.to_csv(options.o+'.csv')    

# Plot functions ------

def plot_roc_and_precision_recall_curves(models, kind_of_data, path_data, n_reads, path_paf, trimming, 
	full_counting, ks, format_file, species):
	"""
	Plots the ROC and the precision-recall-curve for a set of models and a specific data input. 
	- models: a list of trained model. 
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
	- format_file: The format of the input data: can be auto, fasta or fastq. 'auto' should be used by default. 
	- species: the name of the species to be plotted in the plots.
	"""
	if kind_of_data == 'experimental':
		sequences = read_experimental_data(path = path_data, trimming = trimming, n_reads = n_reads, format_file = format_file)
	elif kind_of_data == 'annotation':
		sequences = read_annotation_data(path = path_data, trimming = trimming, n_reads = n_reads, format_file = format_file)
	elif kind_of_data == 'mapped':
		sequences = read_mapped_data(path = path_data, trimming = trimming, n_reads = n_reads, format_file = format_file)
	if path_paf:
		data, labels = prepare_data(sequences, 'mixed', full_counting, ks, False, path_paf, True)
	else:
		data, labels = prepare_data(sequences, 'forwarded', full_counting, ks, False, path_paf, True)
	for model_name in models:
		model = load_model(model_name)
		prediction = model.predict(data.values)
		fpr_grd, tpr_grd, _ = roc_curve(labels, prediction)
		plt.plot(fpr_grd, tpr_grd, label = model_name.split('/')[-1])
	plt.xlabel('False positive rate', fontsize = 17)
	plt.ylabel('True positive rate', fontsize = 17)
	plt.tick_params(axis='both', which='major', labelsize=14)
	plt.title(species + ' cDNA orientation prediction', fontsize = 18)
	plt.legend(fontsize = 15)
	plt.tight_layout()
	plt.savefig('plots/ROC_' + species + ' cDNA orientation prediction.png', dpi = 200)
	plt.close('all')
	for model_name in models:
		model = load_model(model_name)
		prediction = model.predict(data.values)
		precision, recall, _ = precision_recall_curve(labels, prediction)
		plt.plot(recall[:-2], precision[:-2], label = model_name.split('/')[-1])
	plt.xlabel('precision', fontsize = 17)
	plt.ylabel('Recall', fontsize = 17)
	plt.tick_params(axis='both', which='major', labelsize=14)
	plt.title(species + ' cDNA orientation prediction', fontsize = 15)
	plt.legend(fontsize = 15)
	plt.tight_layout()
	plt.savefig('plots/PRC_'+species + ' cDNA orientation prediction.png', dpi = 200)
	plt.close('all')
	return precision, recall, _

def analyze_clusters(path, model=False, csv=False):
	if model:
		model = load_model(model)
	else:
		prediction_csv = pandas.read_csv(csv, index_col = 0, usecols=['id', 'strand', 'predictions'])
	results = []
	lengths = []
	for file in os.listdir(path):
		if not file.startswith('.') and file.endswith('.fq'):
			full_path = path+'/'+file
			sequences = read_experimental_data(full_path, format_file = 'auto' ,trimming = False, gzip_encoded = 'auto', n_reads = int(10e10))
			data, labels = prepare_data(sequences, 'unknown', True, 5, False, False,True)
			try:
				predictons = model.predict(data.values).round()
			except:
				data, labels = prepare_data(sequences, 'unknown', True, 5, False, False,True)
				predictons = model.predict(data.values).round()
			agreement = stats.mode(predictons)[1][0][0]/len(predictons)
			results.append(agreement)
			lengths.append(len(predictons))
	return results, lengths

def analyze_clustersv2(path_clusters = '/Users/angelruiz/Desktop/clusters/clusters/hs/final_clusters.csv',
					   path_prediction = '/Users/angelruiz/Downloads/predictions_cDNA_human_Hs_transcriptome_cnn.csv',
					   paf_file = '/Users/angelruiz/Desktop/RE_data/cdna_human_no_secondary_mapq_60_unique.paf',
					   species = 'hs', prediction_id_col = 0, prediction_prediction_col = 4):
	
    clusters = pandas.read_table(path_clusters, names = ['cluster', 'id'], index_col=1)
    clusters.index = clusters.index.str[0:36]
    clusters.index = clusters.index.str[0:36] # change this to capture the identifier
    #clusters.index = clusters.index.str.split('_').str[0] 
    predictions = pandas.read_csv(path_prediction, usecols=[prediction_id_col, prediction_prediction_col], names = ['id', 'prediction'], index_col=[0], skiprows=1)
    predictions = predictions.prediction.round().replace([1.0, 0], ['-', '+'])
    labels = pandas.read_table(paf_file, usecols = [0,4], index_col = 0, header = None, names = ['id', 'strand'])

    data  = clusters.join(predictions).join(labels)
    data['ncorrect'] = data['prediction']==data['strand']

    cluster_analysis =  pandas.DataFrame(data.groupby('cluster')['ncorrect'].sum())
    cluster_analysis['size'] = data.groupby('cluster').size()

    cluster_analysis['per_correct'] = cluster_analysis['ncorrect']/cluster_analysis['size']
    cluster_analysis['correct_cluster'] = cluster_analysis['ncorrect']>=cluster_analysis['size']//2

    cluster_analysis[cluster_analysis['size'] > 2]['per_correct'].hist()
    plt.title('Accuracy per cluster (Size > 2)')
    plt.savefig(species+'_acc_clusters_gt_2.png')
    plt.close('all')

    cluster_analysis[cluster_analysis['size'] > 1]['per_correct'].hist()
    plt.title('Accuracy per cluster (Size > 1)')
    plt.savefig(species+'_acc_clusters_gt_1.png')
    plt.close('all')

    RE_lazy = labels[labels['strand'] == '+'].size/labels.size
    RE_correct = sum(cluster_analysis['ncorrect'])/sum(cluster_analysis['size'])
    RE_correct_voting = sum(cluster_analysis['size'] * cluster_analysis.correct_cluster)/sum(cluster_analysis['size'])

    print('Lazy model accuracy: ', RE_lazy)
    print('RE accuracy:', RE_correct)
    print('RE plus clustering accuracy:', RE_correct_voting)
    return(0)

if __name__ == '__main__':
	time0 = time.time() 
	if options.train:
		print('\n----Starting Training Pipeline----\n')
		model, history ,data, labels = build_kmer_model(options.s, options.d, options.r, options.a, options.t, 
			True, options.k, options.v, options.e ,options.o, options.use_all_annotation, options.fk,options.ra, options.rm,
			options.oh)

	elif options.test:
		print('\n----Starting Testing Pipeline----\n')
		model = load_model(options.m)
		test_model(model, options.s, options.d, options.r, options.a, options.t, True, options.k, options.oh, False,options.rm)

	elif options.predict:
		print('\n----Starting Prediction Pipeline----\n')
		model = load_model(options.m)
		print('Model successfully loaded')
		print(model.summary())
		make_predictions(model, options.s, options.d, options.r, options.a, options.t, True, options.k, options.oh)
		print('Predictions saved to:', options.o+'.csv')
	time1 = time.time()
	delta = time1 - time0
	print('Elapsed time\t', delta, ' seconds')