# ReorientExpress [![DOI](https://zenodo.org/badge/162119902.svg)](https://zenodo.org/badge/latestdoi/162119902)

ReorientExpress is a program to create, test and apply models to predict the 5'-to-3' orientation of long-reads from cDNA sequencing with Nanopore or PacBio using deep neural networks for samples without a genome or a transcriptome reference. For details on the benchmarkings and analyses performed with this program, please see our publication: https://www.ncbi.nlm.nih.gov/pubmed/31783882

----------------------------
# Table of Contents
----------------------------

   * [Overview](#overview)
   * [Installation](#installation)
   * [Commands and options](#commands-and-options)
   * [Inputs and Outputs](#inputs-and-outputs)
   * [Usage example](#usage-example)
   * [How to cite ReorientExpress](#how-to-cite-ReorientExpress)   
----------------------------
# Overview
----------------------------

ReorientExpress is a tool to predict the orientation of cDNA reads from error-prone long-read sequencing technologies. It was developed with the aim to orientate nanopore long-reads from unstranded cDNA libraries without the need of a genome or transcriptome reference, but it is applicable to any set of long-reads. ReorientExpress implements two Deep Neural Network models: a Multi-Layer Perceptron (MLP) and a Convolutional Neural Network (CNN), and it uses as training input a transcriptome annotation from any species or any other fasta/fasq file of RNA/cDNA sequences for which the orientation is known. 
Training or testing data can thus be experimental data, annotation data or also mapped reads (providing the corresponding PAF file). 
ReorientExpress has three main modes:
- Training a model.
- Testing a model.
- Using a model to orientate input sequences.

These are implemented in three options: train, test and predict. In train mode, the input data is randomly split into three subsets: training, validation and test, with relative proportions of 0.75, 0.125 and 0.125, respectively. The training set is used to train the weights of the DNN model, the validation set is used to optimize the weights during the training process, and the test set has never been seen for training and is only used at the end to evaluate the accuracy of the model.

  


----------------------------
# Installation
----------------------------

ReorientExpress has been developed in Python 3.6. It can be directly cloned and used or installed for an easier reuse and dependency management. 

Currently, you can use pip to do an authomatic installation:
```
pip3 install reorientexpress
```

If some dependencies are not correctly downloaded and installed, using the following can fix it:

```
pip3 install -r requirements.txt
pip3 install reorientexpress
```
Once the package is installed, ReorientExpress can be used from the command line as any other program.

If you want to ensure you have the latest version, we recommend cloning the repository instead, althought you will have to manage the dependencies yourself.

----------------------------
# Commands and options
----------------------------

Once the package is installed it can be used as an independent program. ReorientExpress has three main functions, one of them must be provided when calling the program:

* -train: takes an input an uses it to train a model.
* -test: takes a model and a labeled input and estimate the accuracy of the model using the labeled input data.
* -predict: takes a model and an input and outputs all the sequences in the predicted 5'-to3' orientation. It also gives a certainty score per input sequence.

The different options available for MLP (reorientexpress.py) are:

* **-h, --help**:             Shows a help message with all the options.

*  **-train**:                Set true to train a model.

*  **-test**:                 Set true to test a model.

*  **-predict**:              Set true to use a model to make predictions

*  **-data D, --d D**:        The path to the input data. Must be either fasta or
                        fastq. Can be compressed in gz format. Mandatory.
                        
*  **-output_fastq**:         Set true to get output in fastq format for input fastq file   
              
*  **-source {annotation,experimental,mapped}, --s {annotation,experimental,mapped}**:
                        The source of the data. Must be either 'experimental',
                        'annotation' or 'mapped'. Choose experimental for
                        experiments like RNA-direct, annotation for
                        transcriptomes or other references and mapped for reads mapped 
                        to a reference transcriptome.
                        Mapped reads must be in PAF format to extract the orientation. 
                        Mandatory.
                        
*  **-format {fasta,fastq,auto}, --f {fasta,fastq,auto}**:
                        The format of the input data. Auto by deafult. Change
                        only if inconsistencies in the name.
                        
*  **-annotation A, --a A**:  Path to the PAF file if a mapped training set is used.
                        
*  **-use_all_annotation, -aa**:
                        Uses all the reads from the annotation, instead of only keeping
                        protein_coding, lincRNA, processed_transcript, antisense, and retained_intron. 
                        Use it also if the fasta has unconventional format and gives errors.
                        
*  **-kmers K, --k K**:       The maximum length of the kmers used for training,
                              testing and using the models. It will use from k=1 up to this number.
                        
*  **-reads R, --r R**:       Number of reads to use from the dataset.

*  **-trimming T, --t T**:    Number of nucleotides to trimm at each side. 0 by default.

*  **-reverse_all**:          Reverse-complement all input sequences to double up the training input, 
                        instead of reverse-complementing just a random half of the input sequences, 
                        which is the default. 
                        
*  **-verbose, --v**:         Flag to print detailed information about the 
                        training process.
                        
*  **-epochs E, --e E**:      Number of epochs to train the model.

*  **-output O, --o O**:      Where to store the outputs. using "--train" outputs a
                        model, while using "-predict" outputs a csv.
                        Corresponding extensions will be added.

*  **-model M, --m M**:       The model to test or to predict with.

The different option available for CNN (reoreintexpress-cnn.py) are:


* **-h, --help**:             Shows a help message with all the options.

*  **-train**:                Set true to train a model.

*  **-test**:                 Set true to test a model.

*  **-predict**:              Set true to use a model to make predictions

*  **-output_fastq**:         Set true to get output in fastq format for input fastq file  

*  **-data D, --d D**:        The path to the input data. Must be either fasta or
                        fastq. Can be compressed in gz format. Mandatory.
                        
*  **-source {annotation,experimental,mapped}, --s {annotation,experimental,mapped}**:
                        The source of the data. Must be either 'experimental',
                        'annotation' or 'mapped'. Choose experimental for
                        experiments like RNA-direct, annotation for
                        transcriptomes or other references and mapped for reads mapped 
                        to a reference transcriptome.
                        Mapped reads must be in PAF format to extract the orientation. 
                        Mandatory.
                        
*  **-format {fasta,fastq,auto}, --f {fasta,fastq,auto}**:
                        The format of the input data. Auto by deafult. Change
                        only if inconsistencies in the name.
                        
*  **-annotation A, --a A**:  Path to the PAF file if a mapped training set is used.
                        
*  **-use_all_annotation, -aa**:
                        Uses all the reads from the annotation, instead of only keeping
                        protein_coding, lincRNA, processed_transcript, antisense, and retained_intron. 
                        Use it also if the fasta has unconventional format and gives errors.
                        
*  **-win_size W, --w W**:       Window size for spliting the sequence.
 
*  **-step_size, --step**:       Overlapping size on the the sliding window.
                        
*  **-reads R, --r R**:       Number of reads to use from the dataset.

*  **-trimming T, --t T**:    Number of nucleotides to trimm at each side. 0 by default.

*  **-reverse_all**:          Reverse-complement all input sequences to double up the training input, 
                        instead of reverse-complementing just a random half of the input sequences, 
                        which is the default. 
                        
*  **-verbose, --v**:         Flag to print detailed information about the 
                        training process.
                        
*  **-epochs E, --e E**:      Number of epochs to train the model.

*  **-output O, --o O**:      Where to store the outputs. using "--train" outputs a
                        model, while using "-predict" outputs a csv.
                        Corresponding extensions will be added.

*  **-model M, --m M**:       The model to test or to predict with.


----------------------------
# Inputs and Outputs
----------------------------

All the input sequence files can be in fasta or fastq format. They can also be compressed in gz format.

Input sequences can be of three different types, which we call experimental, annotation or mapped, which can be in FASTA or FASTQ formats, either compressed (in .gz format) or uncompressed. 

* Experimental data refers to any kind of long-read data for which the orientation is known, such as direct RNA-seq, and reads are considered to be given in the 5’-to-3’ orientation. 
* Annotation data refers to the transcript sequences from a reference annotation, such as the human transcriptome reference. It also considers all the sequences to be in the right 5’-to-3’ orientation. Annotation data can also include the transcript type, such as protein coding, processed transcript, etc. 
* Mapped data refers to sequencing data, usually cDNA, whose orientation has been annotated by an independent method, e.g. by mapping the reads to a reference. In this case, a PAF file for the mapping, together with the FASTA/FASTQ file, is required.

### Examples of possible inputs:

#### Experimental

<pre>@0e403438-313b-4497-b1c2-2fd3cc685c1d runid=46930771ed1cff73b50bf5c153000aa904eb5c9c read=100 ch=493 sta
rt_time=2017-10-09T18:11:16Z
CCCGGAAAAUGGUGAAGAAAAUUGAAAUCAGCCAGCACGUCCGUUAAGUCACUUGCUUUACCGCGGCAAACCAAGAUGAAGACGAGCUGUGGGAUCUGGCACUA
CUGUGGUUCCAUUGCAUGAACGGGAAGACAGUGGCUGGCGGGUGCCCUGGACGUACAAAUACCACUCCAAUUGUCACGGUAAAGUCCGCCAUCAGAAGACUGAA
GGAGUUGUAGACCAGUAGACGUUCCAUACACAUUGAGACACUACUGGCCUAUAAUAAUUAAAUGGGUUAUUAAUUUAUUUAUGGCUAACAAAUUGUUCCGAGCU
CGUAUUAAACAGAUAUCGAUGUUGUAUUGUUGUAGUAGUAUUGAAGAGCAAAUCCCACCCAUCCUUCCAUCAACAACCUCCCGUUAUUAUACCGUUAUCCCACC
GCCUACCAUCUUCCCAUAAAAUCCAUC
+
$)/*+7B:314:3.,/.6C;4.*'+69-.14:221'%&#"+)'$$%*)'$%&&)*''(+"$&$%)1*.:/0:7522222/--**--*++*/9>/0-&*('%%%)
,+&031=12+(**)#$#$$'&%((-.-4524,,4*+-:.-./(('@7-)5$'%)))3.,)**-),--/*(/0)(%+1.7*+6)+*7:32&'&*,,(/(('.-1/
3.+../)$-/29:66,*-,&.+.8,(#'&&&')1-//.--((%)(111+''&11,2(%&*./,)5..*'*%.0011%$%%#%'-&(-5+,@6>9;'-)5)**%$
#+*,,,15.''%(*)++,,4,---/064'))()($%#%''*-%&'$'##$$)&'+.%+4,(%'*&$/(&''(0(%/',$,.(&)'#,-$$$'-"$$$$&.+%($
"*+$$$$$%$$#0:*'&%&'+#$&$$"</pre>

#### Annotation

<pre>
>ENSMUST00000193812.1|ENSMUSG00000102693.1|OTTMUSG00000049935.1|OTTMUST00000127109.1|4933401J01Rik-201|4933401J01Rik|1070|TEC|
AAGGAAAGAGGATAACACTTGAAATGTAAATAAAGAAAATACCTAATAAAAATAAATAAA
AACATGCTTTCAAAGGAAATAAAAAGTTGGATTCAAAAATTTAACTTTTGCTCATTTGGT
ATAATCAAGGAAAAGACCTTTGCATATAAAATATATTTTGAATAAAATTCAGTGGAAGAA
TGGAATAGAAATATAAGTTTAATGCTAAGTATAAGTACCAGTAAAAGAATAATAAAAAGA
AATATAAGTTGGGTATACAGTTATTTGCCAGCACAAAGCCTTGGGTATGGTTCTTAGCAC
TAAGGAACCAGCCAAATCACCAACAAACAGAGGCATAAGGTTTTAGTGTTTACTATTTGT
ACTTTTGTGGATCATCTTGCCAGCCTGTAGTGCAACCATCTCTAATCCACCACCATGAAG
GGAACTGTGATAATTCACTGGGCTTTTTCTGTGCAAGATGAAAAAAAGCCAGGTGAGGCT
GATTTATGAGTAAGGGATGTGCATTCCTAACTCAAAAATCTGAAATTTGAAATGCCGCCC
</pre>

#### Mapped

Takes a file with the same format as experimental and also a PAF file with the following format:

<pre>
0M1I3M2D4M3D1M1D10M4I11M1D25M1D6M1D10M1D10M
0e04dd74-26bd-47e3-91bf-0e6e97310067	795	2	410	-	ENST00000584828.5|ENSG0000018406
0.10|OTTHUMG00000132868.4|OTTHUMT00000444515.1|ADAP2-209|ADAP2|907|protein_coding|	907	398	
798	344	432	1	NM:i:88	ms:i:336	AS:i:336	nn:i:0	tp:A:P	cm:i:7	s1:i:82
s2:i:67	dv:f:0.1443	cg:Z:4M1I19M2D15M1I8M4I1M1D6M1I29M1D1M2D13M1D5M2I4M1D21M3I28M2I11M3I8M1I13M2I16M
3D12M1I2M3D5M2I16M2I14M4D12M1I9M4I47M2D1M3D24M2I7M1D25M
0e04dd74-26bd-47e3-91bf-0e6e97310067	795	2	405	-	ENST00000585130.5|ENSG0000018406
0.10|OTTHUMG00000132868.4|OTTHUMT00000444510.1|ADAP2-211|ADAP2|2271|nonsense_mediated_decay|	2271	
1366	1762	340	426	0	NM:i:86	ms:i:334	AS:i:334	nn:i:0	tp:A:S	cm:i:6	
s1:i:67	dv:f:0.1427	cg:Z:19M2D15M1I7M1I3M2I6M1I29M1D1M2D13M1D5M2I4M1D21M3I28M2I11M3I8M1I13M2I16M3D12
M1I2M3D5M2I16M2I14M4D12M1I9M4I47M2D1M3D24M2I7M1D25M
0e04dd74-26bd-47e3-91bf-0e6e97310067	795	2	405	-	ENST00000330889.7|ENSG0000018406
0.10|OTTHUMG00000132868.4|OTTHUMT00000256346.1|ADAP2-201|ADAP2|2934|protein_coding|	2934	1446	
1842	340	426	0	NM:i:86	ms:i:334	AS:i:334	nn:i:0	tp:A:S	cm:i:6	s1:i:67
</pre>
You can read more about the paf file format [here](https://github.com/lh3/miniasm/blob/master/PAF.md). 

### Examples of possible outputs:

Depending on the chosen pipeline, the output can be:
* Training: a keras model object, from the class keras.engine.sequential.Sequential (https://keras.io). It is saved as a binary file that be loaded later.
* Testing: there is no file output. Only the results of the accuracy evaluation displayed on the terminal.
* Predicting: outputs a csv file with all the reads in the predicted 5'-to-3' orientation. It contains three columns: the predicted 5'-to-3' sequence (ForwardedSequence) and the model Score and the read orientation. See below an example:

| Index  | ForwardSequence  | Score  | orientation |
|---|---|---|---|
|  0 | ATGTTGAATAGTTCAAGAAAATATGCTTGTCGTTCCCTATTCAGACAAGCGAACGTCTCA  |  0.8915960788726807 | 0 |
|  1 | TTGAGGAGTGATAACAAGGAAAGCCCAAGTGCAAGACAACCACTAGATAGGCTACAACTA  | 0.9746999740600586  | 1 |
|  2 | AAGGCCACCATTGCTCTATTGTTGCTAAGTGGTGGGACGTATGCCTATTTATCAAGAAAA  |  0.9779879450798035 | 0 |

*Note*: '0' orientation represents '+' and '1' orientation represents '-'. However, the '-' reads are reverses-complemented and provided in the 'ForwardSequence' column.

----------------------------
# Usage example
----------------------------
**Note:** The below commands are for MLP model. Similar commands can be used for CNN model will the replacement of *reorientexpress.py* with *reoreintexpress-cnn.py*

To train a model:

```
reorientexpress.py -train -data path_to_data -source annotation --v -output my_model
```

This trains a model with the data stored in path_to_data, which is an annotation file, suchs as a transcriptome and outputs a file called my_model.model which can be later used to make predictions. Prints relevant information.

Example on test_case provided in the repo:

```
reorientexpress.py -train -data ./test_case/annotation/gencode.vM19.transcripts_50k.fa -source annotation --v -output my_model
```
or 

```
reorientexpress-cnn.py -train -data ./test_case/annotation/gencode.vM19.transcripts_50k.fa -source annotation --v -output my_model
```

To make predictions:

```
reorientexpress.py -predict -data path_to_data -source experimental -model path_to_model -output my_predictions
```
or
```
reorientexpress.py -output_fastq -predict -data path_to_data -source experimental -model path_to_model -output my_predictions
```

This takes the experimental data stored in path_to_data and the model stored in path_to_model and predicts the 5'-to-3' orientation of reads, i.e. converts to forward reads the reads that the model predicts are reverse complemented, printing the results in my_predictions.csv. The output format is same as provided in the 'Examples of possible outputs section above'

In the saved_models/ folder we provide a model trained with the human transcriptome annotation and a model trained with the Saccharomyces cerevisiae transcriptome annoation. They can be directly used with the "-model" flag.

Example on test_case provided in the repo:

```
reorientexpress.py -predict -data ./test_case/experimental/Hopkins_Run1_20171011_1D.pass.dedup_60_unique_50k.fastq -model ./saved_models/Hs_transcriptome_mlp.model -source experimental -output my_predictions
```
or 

```
reorientexpress-cnn.py -predict -data ./test_case/experimental/Hopkins_Run1_20171011_1D.pass.dedup_60_unique_50k.fastq -model ./saved_models/Hs_transcriptome_CNN.model -source experimental -output my_predictions
```
or

```
reorientexpress-cnn.py -output_fastq -predict -data ./test_case/experimental/Hopkins_Run1_20171011_1D.pass.dedup_60_unique_50k.fastq -model ./saved_models/Hs_transcriptome_CNN.model -source experimental -output my_predictions
```


To test the accuracy of the model:

```
reorientexpress.py -test -data path_to_data -annotation path_of_paf_file -source mapped -model path_to_model 
```
Example on test_case provided in the repo:

```
reorientexpress.py -test -data ./test_case/mapped/Hopkins_Run1_20171011_1D.pass.dedup_60_unique_2000.fastq -annotation ./test_case/mapped/cdna_human_no_secondary_mapq_60_unique_2000.paf -model ./saved_models/Hs_transcriptome_mlp.model -source mapped
```

or 

```
reorientexpress-cnn.py -test -data ./test_case/mapped/Hopkins_Run1_20171011_1D.pass.dedup_60_unique_2000.fastq -annotation ./test_case/mapped/cdna_human_no_secondary_mapq_60_unique_2000.paf -model ./saved_models/Hs_transcriptome_CNN.model -source mapped
```

The ouput accuracy (precision, recall, F1-score, support) will be displayed on the screen. 

----------------------------
# How to cite ReorientExpress
----------------------------
Ruiz-Reche A, Srivastava A, Indi JA, de la Rubia I, Eyras E. ReorientExpress: 
reference-free orientation of nanopore cDNA reads with deep learning. Genome
Biol. 2019 Nov 29;20(1):260. doi: 10.1186/s13059-019-1884-z. https://www.ncbi.nlm.nih.gov/pubmed/31783882




