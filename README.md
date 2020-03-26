# TALE
Transformer-based protein function Annotation with joint feature-Label Embedding


![TALE Architecture](/ProteinFuncPred.png)

## Dependencies
* TensorFlow >=1.13
* Download [Diamond](http://www.diamondsearch.org/index.php) and put the executable file into TALE/diamond/


## For users
### If you want to use TALE+ for prediction, prepare your seqeunces file in the fasta format and run:
`python predict.py --input_seq $path_to_your_fasta_file --ontology on --outputpath $path_to_your_output_file`

where on=mf,bp,cc for MFO,BPO and CCO, respectively.

## For developers
### Training and test data:
* Data/Gene_Ontology/EXP_Swiss_Prot/train_seq_mf: The training sequence file for MFO 
* Data/Gene_Ontology/EXP_Swiss_Prot/train_label_mf: The training label file for MFO
* Data/Gene_Ontology/EXP_Swiss_Prot/test_seq_mf: The test sequence file for MFO
* Data/Gene_Ontology/EXP_Swiss_Prot/test_label_mf: The test label file for MFO
* Data/Gene_Ontology/EXP_Swiss_Prot/mf_on_1.pickle: The ontology file for MFO

### Data formats:
* The sequence file is a **list**, where each element is a directory having the follo 
#### Label file
* The label file is a **list**, where each element is a list containing the indexes of labels (GO terms).
#### Ontology file:
* The ontology file is a directory, where each key is a GO term (e.g. 'GO:0030234') in the ontology. Each value is also a directory containing the information for that key:
..* 'name': The name of the GO term
..* 'ind':  The index of this GO term
..* 'father': The parent GO terms
..* 'child': The children GO terms

### Training:

### Trained models:


## Citation
TBD
