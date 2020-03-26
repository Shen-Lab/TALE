# TALE
Transformer-based protein function Annotation with joint feature-Label Embedding


![TALE Architecture](/ProteinFuncPred.png)

## Dependencies
* TensorFlow >=1.13
* Download [Diamond](http://www.diamondsearch.org/index.php) and put the executable file into TALE/diamond/


## For users
### If you want to use TALE+ for prediction, prepare your seqeunces file in the fasta format and run:
`python predict.py --input_seq $your_fasta_file --ontology on`

where on=mf,bp,cc for MFO,BPO and CCO, respectively.

## For developers
### File descriptions:
* Data/Gene_Ontology/EXP_Swiss_Prot/train_seq_mf: The training sequence file for MFO 
* Data/Gene_Ontology/EXP_Swiss_Prot/train_label_mf: The training label file for MFO
* Data/Gene_Ontology/EXP_Swiss_Prot/test_seq_mf: The test sequence file for MFO
* Data/Gene_Ontology/EXP_Swiss_Prot/test_label_mf: The test label file for MFO
* Data/Gene_Ontology/EXP_Swiss_Prot/mf_on_1.pickle: The ontology file for MFO

### File formats:
#### Sequence file:
#### Label file
#### Ontology file:

### Training:

### Evalutation:


## Citation
TBD
