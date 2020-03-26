# TALE
Transformer-based protein function Annotation with joint feature-Label Embedding


![TALE Architecture](/ProteinFuncPred.png)

## Dependencies
* TensorFlow >=1.13
* Download [Diamond](http://www.diamondsearch.org/index.php) and put the executable file into TALE/diamond/

## File explanation
### Data/Gene_Ontology/EXP_Swiss_Prot/
* train_seq_mf: The training sequence file for MFO
* train_label_mf: The training label file for MFO
* test_seq_mf:


## For users:
### If you want to use TALE+ for prediction, prepare your seqeunces file in the fasta format and run:
Inline `python predict.py --input_seq $your_fasta_file --ontology $on`
where on=mf,bp,cc for MFO,BPO and CCP, respectively.




## Citation
TBD
