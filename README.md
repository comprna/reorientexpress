# ReorientExpress

This is a script that is used to create, test and use models to predict the orientation of cDNA reads using deep neural networks.

----------------------------
# Table of Contents
----------------------------

   * [Overview](#overview)
   * [Installation](#installation)
   * [Commands and options](#commands-and-options)
   * [Inputs and Outputs](#inputs-and-outputs)
   * [Usage example](#usage-example)
   
----------------------------
# Overview
----------------------------

ReorientExpress is a tool which main purpose is to predict the orientation of cDNA or RNA-direct reads. It builds kmer-based models using Deep Neural Networks and taking as input a transcriptome annotation or any other fasta/fasq file for which the sequences orientation is known. 
The software can work with experimental data, annotation data and also with mapped reads providing the corresponding PAF file. 
ReorientExpress has three main utilities:
- Training a model.
- Testing a model.
- Using a model to orient de input sequences.


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

----------------------------
# Commands and options
----------------------------

Once the package is installed it can be used as an independent program. ReorientExpress has three main functions, one of them must be provided when calling the program:

* -train: takes an input an uses it to train a model.
* -test: takes a model and a labeled input and tries the performance of the model in the input.
* -predict: takes a model and an input outputs all the sequences in forward. It also gives a certainty score. 

The different options that the program takes are:

* **-h, --help**:            Shows a help message with all the options.

*  **-train**:            Set true to train a model.

*  **-test**:                 Set true to test a model.

*  **-predict**:              Set true to use a model to make predictions

*  **-data D, --d D**:        The path to the input data. Must be either fasta or
                        fastq. Can be compressed in gz. Mandatory.
                        
*  **-source {annotation,experimental,mapped}, --s {annotation,experimental,mapped}**:
                        The source of the data. Must be either 'experimental',
                        'annotation' or 'mapped'. Choose experimental for
                        experiments like RNA-direct, annotation for
                        transcriptomes or other references and mapped for mapped
                        cDNA reads. Mapped reads require a paf file to know the
                        orientation. Mandatory.
                        
*  **-format {fasta,fastq,auto}, --f {fasta,fastq,auto}**:
                        The format of the input data. Auto by deafult. Change
                        only if inconsistencies in the name.
                        
*  **-annotation A, --a A**:  Path to the paf file if a mapped training set which
                        requires a paf reference is being used.
                        
*  **-use_all_annotation, -aa**:
                        Uses all the reads, instead of only keeping
                        antisense,lincRNA,processed_transcript,
                        protein_coding, and retained_intron. Use it also if
                        the fasta has unconventional format and gives errors.
                        
*  **-kmers K, --k K**:       The maximum length of the kmers used for training,
                        testing and using the models.
                        
*  **-reads R, --r R**:       Number of reads to read from the dataset.

*  **-trimming T, --t T**:    Number of nucleotides to trimm at each side. 0 by
                        default.
                        
*  **-verbose, --v**:         Whether to print detailed information about the
                        training process.
                        
*  **-epochs E, --e E**:      Number of epochs to train the model.

*  **-output O, --o O**:      Where to store the outputs. using "--train" outputs a
                        model, while using "-predict" outputs a csv.
                        Corresponding extensions will be added.

*  **-model M, --m M**:       The model to test or to predict with.

----------------------------
# Inputs and Outputs
----------------------------

Al the input sequence files can be in fasta or fastq format. They can also be compressed in gz. 

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

Takes a file with the same format as experimental and also a paf file with the following shape:

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
* Training: a keras model object, from the class keras.engine.sequential.Sequential. It's saved as a binary file that be loaded later.
* Testing: there is no file output. Only the results displayed on the terminal.
* Predicting: outputs a csv file with all the reads in forward orientation. It contains three columns: the index, the Forwarded Sequence and the model Score. See below an example:

| Index  | ForwardSequence  | Score  |
|---|---|---|
|  0 | ATGTTGAATAGTTCAAGAAAATATGCTTGTCGTTCCCTATTCAGACAAGCGAACGTCTCA  |  0.8915960788726807 |
|  1 | TTGAGGAGTGATAACAAGGAAAGCCCAAGTGCAAGACAACCACTAGATAGGCTACAACTA  | 0.9746999740600586  |
|  2 | AAGGCCACCATTGCTCTATTGTTGCTAAGTGGTGGGACGTATGCCTATTTATCAAGAAAA  |  0.9779879450798035 |

----------------------------
# Usage example
----------------------------

To train a model:

```
reorientexpress -train -data path_to_data -source annotation --v -output my_model
```

Which train a model with the data stored in path_to_data, which is an annotation file, suchs as a transcriptome and outputs a file called my_model.model which can be later used to make predictions. Prints relevant information.

To make predictions:

```
reorientexpress -predict -data path_to_data -source experimental -model path_to_model -output my_predictions
```

Which takes the experimental data stored in path_to_data and the model stored in path_to_model and converts to forward reads the reads that the model predicts are in reverse complementary, printing the results in my_predictions.csv. 
Also, in the saved_models/ folder we provide a model trained with the human transcriptome annotation and a model trained with the Saccharomyces cerevisiae transcriptome annoation. They can be directly used with the "-model" flag.


