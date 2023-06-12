######################################################
# get the sequence embedding vertors / matrices
# by SZ; 6/9/23
######################################################

import tensorflow as tf
import numpy as np
import train
import sys
import pickle
import argparse
sys.path.insert(0, 'Utils/')
import hparam
import amino_acid
import metric
import os
import math

def dict_save(dictionary, path):
	with open(path,'wb') as handle:
		pickle.dump(dictionary, handle)
	return 0

def main():
	parser = argparse.ArgumentParser(description='Arguments for predict.py')
	parser.add_argument('--trained_model', default=None, nargs='*', type=str)
	parser.add_argument('--on', default=None, type=str)
	parser.add_argument('--fasta', default=None, type=str)
	parser.add_argument('--out', default='./emb_dict.pkl', type=str)
	parser.add_argument('--data', default='../data/ours/', type=str)      

	parser.add_argument('--batch_size', default=1, type=int)
 
	args = parser.parse_args()

	### load the inference sequences
	if args.fasta is None:
		raise ValueError("Must provide the input fasta sequences.")
		quit()

	seq={}
	s=''
	k=''
	with open(args.fasta, "r") as f:
		for lines in f:
		    if lines[0]==">":
		         if s!='':
		              seq[k] = s
		              s=''
		         k = lines[1:].strip('\n')
		    else:
		         s+=lines.strip('\n')
	seq[k] = s	

	### for the models (prepare a list of the models for emsembling)
	if args.trained_model is None:
		# use default models
		if args.on=='mf':
			args.trained_model = ['../trained_model/Our_model1_MFO', '../trained_model/Our_model2_MFO', '../trained_model/Our_model3_MFO']
		
		elif args.on=='bp':
			args.trained_model = ['../trained_model/Our_model1_BPO', '../trained_model/Our_model2_BPO', '../trained_model/Our_model3_BPO']
		
		elif args.on=='cc':
			args.trained_model = ['../trained_model/Our_model1_CCO', '../trained_model/Our_model2_CCO', '../trained_model/Our_model3_CCO']
	
	### embedding	
	seq_emb_list = []
	final_emb_list = []
 
	for model_idx, model in enumerate(args.trained_model):	
		seq_emb, final_emb = seq_embedding(model, seq, args.data, batch_size = args.batch_size)

		seq_emb_list.append(seq_emb)
		final_emb_list.append(final_emb)

	out_dict = {}
	out_dict['seq_emb'] = (np.array(seq_emb_list)).mean(axis=0)

	if final_emb_list[0] is not None:
		out_dict['final_emb'] = (np.array(final_emb_list)).mean(axis=0)
	else:
		out_dict['final_emb'] = None
 
	_ = dict_save(out_dict, args.out)


def seq_embedding(model_path, seq_model, data_path, batch_size=None):
	# get the sequences embedding through trained deep learning  model
	# argvs: model_path: the path to the trained model
	#	 seq_model: a dic stores the input sequences
	# output:  a probability matrix with shape (len(seq_model), # GO_terms].
	
	tf.keras.backend.clear_session()

	### hyper-parameters
	with open(model_path+".hparam", "rb") as f:
		hparams=pickle.load(f)
	hparams['train']=False
	hparams['data_path'] = data_path

	if batch_size is not None:
		hparams['batch_size']=batch_size
	else:
		hparams['batch_size']=1
		batch_size = 1

	### sequence transformation 
	test_x=[]
	for i in seq_model:
		test_x.append(amino_acid.to_int(seq_model[i], hparams))
	print ("predicting #seq:", len(test_x))

	### prepare the model
	if model_path is None:
		raise ValueError("Must specify a model to evaluate.")
		return None
	print ("start evaluating model: "+model_path)

	model1 = train.HMC_models(hparams)
	with tf.device('/gpu:0'):
		holder_list = model1.Main_model()  #------------holder_list: [model_input, model_output, loss]

	val_list=[v for v in tf.global_variables()]
	saver = tf.train.Saver(val_list)

	### inference
	config = tf.compat.v1.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.8
	config.gpu_options.allow_growth = True

	with tf.Session(config = config) as sess:
		saver.restore(sess, model_path)

		seq_emb_all = []
		fianl_emb_all = []

		iterations = math.ceil(len(test_x) / batch_size)

		for ite in range(iterations):
			x = test_x[ite*batch_size: (ite+1)*batch_size]
			
			seq_emb, final_emb = sess.run([holder_list[5], holder_list[6]], {holder_list[0]: x})
			seq_emb = np.array(seq_emb)
			seq_emb_all.extend(seq_emb)
			print('Seq embedding shape:', seq_emb.shape)

			if final_emb is not None:
				final_emb = np.array(final_emb)
				fianl_emb_all.extend(final_emb)
			else:
				fianl_emb_all = None
	
			print(f'iterations: {ite}/{iterations}')
	
	#  evaluate  the rest test samples after batch evaluation
	if fianl_emb_all is not None:
		fianl_emb_all = np.array(fianl_emb_all)

	return np.array(seq_emb_all), fianl_emb_all


if __name__=='__main__':
	main()
