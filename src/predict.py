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

#predict GO terms based on fasta results.
def main():
        parser = argparse.ArgumentParser(description='Arguments for predict.py')
        parser.add_argument('--trained_model', default=None, type=str)
        parser.add_argument('--on', default=None, type=str)
        parser.add_argument('--data', default='../data/Gene_Ontology/tf_version/', type=str)
        parser.add_argument('--out', default='../Resultours/output1', type=str)
        args = parser.parse_args()

        if args.trained_model==None:
                raise ValueError("Must specify a model to evaluate.")
        #with open("../data/Gene_Ontology/tf_version/test_label_"+args.on, "rb") as f:
        #     label = pickle.load(f)

        #with open("../data/Gene_Ontology/tf_version/"+args.on+"_go_1.pickle", "rb") as f:
        #    graph = pickle.load(f)
        #    nlabels = len(graph)
        #    del graph

        #true_label=[]

        #for i in label:
        #       true_label1 = np.zeros(nlabels, dtype=np.int32)
        #       true_label1[i] = 1
        #       true_label.append(true_label1)
        #true_label = np.array(true_label) 
        #print (true_label.shape)

        #truelabel = np.load("../Resultours/truelabel_"+args.on+".npy")
        #print (truelabel.shape)
        #print (np.array_equal(true_label, truelabel))

        with open(args.data+"/test_seq_"+args.on, "rb") as f:
              seq = pickle.load(f)
        preds = predict_trainedmodel(args.trained_model, seq)
        np.save(args.out, preds)


def predict_trainedmodel(model_path, seq_model, batch_size=None):
	# predict GO terms through trained deep learning  model
	# argvs: model_path: the path to the trained model
	#	 seq_model: a list stores the input sequences
	# output:  a probability matrix with shape (len(seq_model), # GO_terms].
	
	tf.keras.backend.clear_session()
	if model_path==None:
		raise ValueError("Must specify a model to evaluate.")


	with open(model_path+".hparam", "rb") as f:
		hparams=pickle.load(f)

	if batch_size != None:
		hparams['batch_size']=batch_size
	test_x=[]
	for i in range(len(seq_model)):
		test_x.append(amino_acid.to_int(seq_model[i]['seq'], hparams))
	print ("predicting #seq:", len(test_x))

	print ("start evaluating model: "+model_path)


	def sparse_to_dense(y,  length):
		out=np.zeros((len(y), length), dtype=np.int32)
		for i in range(len(y)):
			#print (y[i])
			for j in y[i]:
				out[i][j]=1
		
		return out


	#hparams['train']=True

	model1 = train.HMC_models(hparams)
	with tf.device('/gpu:0'):
			holder_list = model1.Main_model()  #------------holder_list: [model_input, model_output, loss]
			#optimizer = tf.train.AdamOptimizer(learning_rate=hparams['lr'])
			#train_op = optimizer.minimize(holder_list[2])
			#init_op = tf.global_variables_initializer()

	val_list=[v for v in tf.global_variables()]
	saver = tf.train.Saver(val_list)
	batch_size = hparams['batch_size']
	with tf.Session() as sess:
		saver.restore(sess, model_path)

		pred_scores=[]
		iterations = int(len(test_x) // hparams['batch_size'])
		for ite in range(iterations):
			x= test_x[ite*batch_size: (ite+1)*batch_size]
			
			pred_score = sess.run(holder_list[3], {holder_list[0]: x})
			print (np.array(pred_score).shape )	
			pred_scores.extend(pred_score)
			print (f'iterations: {ite}/{iterations}')
	
	#  evaluate  the rest test samples after batch evaluation
	tf.keras.backend.clear_session()

	rest_init = iterations*batch_size
	batch_size = hparams['batch_size'] = 1
	model1 = train.HMC_models(hparams)
	with tf.device('/gpu:0'):
		holder_list = model1.Main_model()
		#optimizer = tf.train.AdamOptimizer(learning_rate=hparams['lr'])
		#train_op = optimizer.minimize(holder_list[2])
		#init_op = tf.global_variables_initializer()
	val_list=[v for v in tf.global_variables()]
	saver = tf.train.Saver(val_list)
	with tf.Session() as sess:
		saver.restore(sess, model_path)
		for ite in range(rest_init, len(test_x)):
			x= test_x[ite:ite+1]
			
			pred_score= sess.run(holder_list[3], {holder_list[0]: x})
			print (np.array(pred_score).shape )

			pred_scores.extend(pred_score)
			print (f'iterations: {ite}/{len(test_x)}')
		

	return np.array(pred_scores)




if __name__=='__main__':
	main()
