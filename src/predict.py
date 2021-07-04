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
def default_trained_model(on):
	trained_models=[]
	if on=='mf':
		trained_models = ['../trained_models/app/mf/ckpt_epoch_180', '../trained_models/app/mf_1/ckpt_epoch_180', '../trained_models/app/mf_10/ckpt_epoch_180']

	elif on=='bp':
		trained_models = ['../trained_models/app/bp_0.5/ckpt_epoch_55', '../trained_models/app/bp/ckpt_epoch_55', '../trained_models/app/bp_10/ckpt_epoch_55']

	elif on=='cc':
		trained_models = ['../trained_models/app/cc_1/ckpt_epoch_189', '../trained_models/app/cc/ckpt_epoch_189', '../trained_models/app/cc_10/ckpt_epoch_189'] 

	return trained_models

def main():
        parser = argparse.ArgumentParser(description='Arguments for predict.py')
        parser.add_argument('--trained_model', default=None, type=str)
        parser.add_argument('--on', default=None, type=str)
        parser.add_argument('--fasta', default=None, type=str)
        parser.add_argument('--out', default='../Resultours/output1', type=str)
        parser.add_argument('--data', default='../data/Gene_Ontology/app_version/', type=str)
       
        args = parser.parse_args()

        args.traindnmd = args.data + "/trainseq_"+args.on+".fasta.dmnd"

        if args.fasta == None:
                raise ValueError("Must provide the input fasta sequences.")

        if args.trained_model==None:
                # use default models
                args.trained_model = default_trained_model(args.on)
        else:
                args.trained_model = [args.trained_model]


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
                             
	
        preds_tale_list=[]
        for model in args.trained_model:	
             preds_tale_list.append( predict_trainedmodel(model, seq))
        preds_tale =  (np.array(preds_tale_list)).mean(axis=0)
        print ((np.array(preds_tale_list)).shape, preds_tale.shape)
        preds_diamond = predict_diamond(args.fasta, args.traindnmd, seq, len(preds_tale[0]), args.on, args.data)
        
        optimal_a = {'mf': 0.4, 'bp':0.5, 'cc':0.6}
        optimal_thres={'mf':0.3, 'bp': 0.09 , 'cc': 0.39}   
        preds=[]
        for i,j in zip(preds_diamond, preds_tale): 
            if np.sum(i)!=0:
              preds.append(i* (1-optimal_a[args.on]) + j * optimal_a[args.on])
            else:
              preds.append(j)


        with open(args.data+args.on+"_go_1.pickle", "rb") as f:
                on1 = pickle.load(f)

        ind_on={}

        for i in on1:
                ind_on[on1[i]['ind']]=(i, on1[i]['name'])

        with open(args.out, "w") as f:
           i=0
           for s in seq:
 
              sorted_idx = np.argsort(-preds[i])
 
              for k in range(100):
                  f.write("%-40s %-100s %.3f\n" %(s, ind_on[sorted_idx[k]], preds[i][sorted_idx[k]]))     
              
              k=100
              while preds[i][sorted_idx[k]]>=0.5 and k<len(sorted_idx):
                  f.write("%-40s %-100s %.3f\n" %(s, ind_on[sorted_idx[k]], preds[i][sorted_idx[k]]))
                  k+=1

              i+=1
                 
        


def predict_diamond(fasta, train_dnmd, seq, nlabels, on, data_path):

   output = os.popen("../diamond/diamond blastp -d "+train_dnmd+" -q "+fasta+" --outfmt 6 qseqid sseqid bitscore").readlines()

   with open(data_path + "train_seq_"+on, "rb") as f:
       train_seq = pickle.load(f) 


   test_bits={}
   test_train={}
   for lines in output:
       line = lines.strip('\n').split()
       if line[0] in test_bits:
          test_bits[line[0]].append(float(line[2]))
          test_train[line[0]].append(line[1])
       else:
          test_bits[line[0]] = [float(line[2])]
          test_train[line[0]] = [line[1]]
       #print (lines) 



   preds_score=[]
   for s in seq:
           probs = np.zeros(nlabels, dtype=np.float32)
           if s in test_bits:
                weights = np.array(test_bits[s])/np.sum(test_bits[s])

                for j in range(len(test_train[s])):
                  temp = np.zeros(nlabels)
                  #print (s, j, test_train[str(s)])
                  temp[ train_seq[int(test_train[s][j])]['label']  ] = 1.0
                  probs+= weights[j]* temp

           preds_score.append(probs)

   return np.array(preds_score)


def predict_trainedmodel(model_path, seq_model, batch_size=None):
	# predict GO terms through trained deep learning  model
	# argvs: model_path: the path to the trained model
	#	 seq_model: a dic stores the input sequences
	# output:  a probability matrix with shape (len(seq_model), # GO_terms].
	
	tf.keras.backend.clear_session()
	if model_path==None:
		raise ValueError("Must specify a model to evaluate.")


	with open(model_path+".hparam", "rb") as f:
		hparams=pickle.load(f)

	if batch_size != None:
		hparams['batch_size']=batch_size
	test_x=[]
	for i in seq_model:
		test_x.append(amino_acid.to_int(seq_model[i], hparams))
	print ("predicting #seq:", len(test_x))

	print ("start evaluating model: "+model_path)


	def sparse_to_dense(y,  length):
		out=np.zeros((len(y), length), dtype=np.int32)
		for i in range(len(y)):
			#print (y[i])
			for j in y[i]:
				out[i][j]=1
		
		return out


	hparams['train']=False
	hparams['batch_size']=1

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
		iterations = int(len(test_x))
		for ite in range(iterations):
			x= test_x[ite*batch_size: (ite+1)*batch_size]
			
			pred_score = sess.run(holder_list[3], {holder_list[0]: x})
			print (np.array(pred_score).shape )	
			pred_scores.extend(pred_score)
			print (f'iterations: {ite}/{iterations}')
	
	#  evaluate  the rest test samples after batch evaluation

	return np.array(pred_scores)




if __name__=='__main__':
	main()
