# TALE
Transformer-based protein function Annotation with joint sequence-Label Embedding


![TALE Architecture](/ProteinFuncPred.png)


Joint Feature-Label Embedding 

Input feature: sequence data (using transformer) 

Output label: hierarchical nodes on directed graphs

## Dependencies
* TensorFlow >=1.13
* For TALE+ (TALE+Diamond), please download [Diamond](http://www.diamondsearch.org/index.php) and put the executable file into TALE/diamond/


## For users
### If you want to use TALE+ for prediction, prepare your sequence file in the fasta format and go to src/ and run:
`python predict.py --input_seq $path_to_your_fasta_file --ontology on --outputpath $path_to_your_output_file`

where on=mf,bp,cc for MFO,BPO and CCO, respectively.

## For developers
### Our training and test data:
* Data/train_seq_mf: The training sequence file for MFO 
* Data/train_label_mf: The training label file for MFO
* Data/test_seq_mf: The test sequence file for MFO
* Data/test_label_mf: The test label file for MFO
* Data/mf_go_1.pickle: The ontology file for MFO

### Data formats:
#### Sequence
The sequence file is a list, where each element is a directory having the following information:
* 'ID': The ID of the sequence in Swiss-Prot
* 'ac': The acession number of the sequence in Swiss-Prot
* 'date': The date of the sequence released in Swiss-Prot
* 'seq': The amino acid sequence
* 'GO':  The GO annotations of the sequence
#### Label 
* The label file is a list, where each element is a list containing the indexes of labels (GO terms).
#### Ontology 
The ontology file is a directory, where each key is a GO term (e.g. 'GO:0030234') in the ontology. Each value is also a directory containing the information for that key:
* 'name': The name of the GO term
* 'ind':  The index of this GO term
* 'father': The parent GO terms
* 'child': The children GO terms

### Training:
In order to train the model, under src/, run:

`python train.py --batch_size 32 --epochs 100 --lr 1e-3 --save_path ./log/ --ontology mf --data_path ../data/ --regular_lambda 0`

The above example is to train a model with 32 batch size, 100 epochs, 1e-3 learning rate, MFO ontology, 0 lambda value, with training data path at '../data/Gene_Ontology/EXP_Swiss_Prot/' and save the trained model in './log/'.

### Trained models:
The trained models are in 'trained_models/'. (e.g. Our_modelk_MFO* is the kth best model on MFO trained on our dataset; CAFA3_modelk_MFO* is the kth best model on MFO trained on CAFA3 dataset.)


## Citation
TBD


## Contact:
Yang Shen: yshen@tamu.edu

Yue Cao:  cyppsp@tamu.edu
