import tensorflow as tf
import numpy as np
import train
import sys
import pickle
sys.path.insert(0, 'Utils/')
import hparam
import amino_acid
import metric
import os

#predict GO terms based on fasta results.

def data_load_fasta(data_path):

	seq_test=[]
	new_seq={}
	seq_ind={}
	with open(data_path, "r") as f:
		for lines in f:
			line = lines.strip('\n')
			if line[0] == '>':
				
				if 'ID' in new_seq:
					seq_test.append(new_seq)
					seq_ind[seq_test[-1]['ID']]= len(seq_test)-1
				new_seq={}
				new_seq['ID'] = line[1:].split()[0]
				new_seq['seq']=''
			else:
				new_seq['seq']+=line
	seq_test.append(new_seq)	
	seq_ind[new_seq['ID']] = len(seq_test)-1
	print (len(seq_test))
	return seq_test, seq_ind

def data_to_fasta(seq_sp, fasta_path):

	with open(fasta_path, "w") as f:
		for i in range(len(seq_sp)):
			f.write('>'+str(i)+'\n')
			f.write(seq_sp[i]['seq'])
			f.write('\n')

def sparse_to_dense(y,  length):
	out=np.zeros((len(y), length), dtype=np.int32)
	for i in range(len(y)):
		#print (y[i])
		for j in y[i]:
			out[i][j]=1
	return out

 
def diamond_pred(seq_ind, fasta_path, train_data_path, diamond_path, on, cut_num, taxonid):
	# predict GO terms from fasta sequence through diamond (sequence similairty)
	# args:  seq_ind:  a dic stores the mapping from the ID of each fasta sequence to the index in the final output
	#  	 fasta_path: path to the input fasta sequence
	#	 train_data_path: path to the ontology and training data labels
	#        diamond_path: path to the diamond executable problem
	#		on:  ontology (could be mf, bp or cc)
	# output:  the predicted prob for each seqeuence assiociated with each GO term, shape: [#fasta_seqs, #GO terms]

	with open(train_data_path+"/"+on+"_go_"+cut_num+".pickle", "rb") as f:
		go_dag = pickle.load(f)


	with open(train_data_path+"./train_label_"+on, "rb") as f:
		train_label=pickle.load(f)
	print ("training_label", len(train_label))	
	train_label = sparse_to_dense(train_label, len(go_dag))
	os.system(diamond_path+"/diamond blastp -d "+diamond_path+'/train_seq_'+on+" -q "+fasta_path+" --outfmt 6 qseqid sseqid bitscore > ./diamond"+taxonid+".res")
	
	score_dic = {}		

	with open("./diamond"+taxonid+".res", "r") as f:		
		for lines in f:
			line = lines.strip('\n').split()
			ind_test = int(seq_ind[line[0]])
			ind_train = int(line[1])
			if ind_test not in score_dic:
				score_dic[ind_test]={}
				score_dic[ind_test]['sum']= float(line[2])
				score_dic[ind_test]['similar']=[ np.array(train_label[ind_train])*float(line[2]) ]
			else:
				score_dic[ind_test]['sum']+=float(line[2])
				score_dic[ind_test]['similar'].append( np.array(train_label[ind_train])*float(line[2]) )
	scores=[]
	for i in range(len(seq_ind)):
		if i in score_dic:
			arr = np.sum(score_dic[i]['similar'], axis=0)/score_dic[i]['sum']
		else:
			arr = np.zeros(len(go_dag), dtype=np.float32)
		scores.append(arr)
	return scores




def assign_default(on, model_path1, model_path2, model_path3, alpha):
	if model_path1 == None:
		print ("did not find input model. Will use default model.")

		model_path1 = "../trained_models/model1_"+on

		model_path2 = "../trained_models/model2_"+on

		model_path3 = "../trained_models/model3_"+on		

	if alpha==None:
		print ("did not find input alpha. Will use default alpha.")
		if on=='mf':
			alpha=0.4
		elif on=='bp':
			alpha=0.5
		else:
			alpha=0.7

	return model_path1, model_path2, model_path3, alpha

def main():

	flags = tf.flags
	FLAGS = flags.FLAGS
	flags.DEFINE_string("model_path1", None, "path to the model1")
	flags.DEFINE_string("model_path2", 'None', "path to the model2")
	flags.DEFINE_string("model_path3", 'None', "path to the model3")
	flags.DEFINE_string("input_seq", None, "path to the input seq (fasta_format)")
	flags.DEFINE_string("train_data", "../data/Gene_Ontology/EXP_Swiss_Prot/", "path to the training data and ontology data")
	flags.DEFINE_string("outputpath", "./result.txt" , "e")

	flags.DEFINE_float("alpha", None, "alpha value")
	flags.DEFINE_float("batch_size", None, "")
	flags.DEFINE_string('ontology', None, "")
	flags.DEFINE_string('runseed','10090' ,"If you want to run this program simutaneously for more than one time, you need to give each individual run different runseed. This is due to diamond intermediate file.")

	if FLAGS.input_seq==None:
		raise ValueError("Must give a fasta sequence file")

	if FLAGS.ontology==None:
		raise ValueError("Must specify the ontology (mf, bp, cc)")
	
	# assign default values for three models and alpha if they are not defined in user inputs.
	model_path1, model_path2, model_path3, alpha =  assign_default(FLAGS.ontology, FLAGS.model_path1, FLAGS.model_path2, FLAGS.model_path3, FLAGS.alpha)


	with open(model_path1+".hparam", "rb") as f:
		hparams=pickle.load(f)

	hparams['train']=False
	hparams['train_data']=FLAGS.train_data

	with open(hparams['train_data']+"/"+FLAGS.ontology+"_go_"+hparams['cut_num']+".pickle", "rb") as f:
		go_dag = pickle.load(f)
	ind_to_goterm={}
	for go in go_dag:
		ind_to_goterm[go_dag[go]['ind']] = go

	with open(hparams['train_data']+'/mf_go_'+hparams['cut_num']+".pickle", "rb") as f:
		go_dag1 = pickle.load(f)
	with open(hparams['train_data']+'/bp_go_'+hparams['cut_num']+".pickle", "rb") as f:
		go_dag2 = pickle.load(f)
	with open(hparams['train_data']+'/cc_go_'+hparams['cut_num']+".pickle", "rb") as f:
		go_dag3 = pickle.load(f)
	hparams['nb_classes']= len(go_dag1)+len(go_dag2)+len(go_dag3)
	hparams['nb_classes_mf']=len(go_dag1)
	hparams['nb_classes_bp']=len(go_dag2)
	hparams['nb_classes_cc']=len(go_dag3)


	if FLAGS.ontology!=None:
		hparams['ontology'] = FLAGS.ontology
	
	print ("start evaluating model: ")
	print ("model1: "+model_path1)
	print ("model2: "+model_path2)
	print ("model3: "+model_path3)
	print ("ontology:"+hparams['ontology'])
	print ("test data seq:"+FLAGS.input_seq)



	test_seq,_ = data_load_fasta(FLAGS.input_seq)
	

	all_pred_scores=[]

	pred_scores,_ = predict_ensemble(alpha, hparams, \
		model_path=model_path1, seq_model=test_seq, batch_size=FLAGS.batch_size, taxonid=FLAGS.runseed)

	all_pred_scores.append(pred_scores)

	if model_path2!='None':
		pred_scores,_ =predict_ensemble(alpha, hparams,  \
                	model_path=model_path2, seq_model=test_seq, batch_size=FLAGS.batch_size, taxonid=FLAGS.runseed) 	
		all_pred_scores.append(pred_scores)

	if model_path3!='None':
		pred_scores,_ =predict_ensemble(alpha, hparams,  \
			model_path=model_path3, seq_model=test_seq, batch_size=FLAGS.batch_size, taxonid=FLAGS.runseed)
		all_pred_scores.append(pred_scores)	

	scores = np.mean(all_pred_scores, axis=0)

	sorted_index=np.argsort(-scores, axis=1)
	print(sorted_index.shape)

	with open(FLAGS.outputpath, "w") as f:
		for i in range(len(sorted_index)):
			for j in range(1000):
				if scores[i][sorted_index[i][j]]>=0.1:
					f.write("%s    %s    %.3f\n" %(test_seq[i]['ID'], ind_to_goterm[sorted_index[i][j]], scores[i][sorted_index[i][j]]))
				else:
					break

	os.system("rm ./diamond"+FLAGS.runseed+".res")

def predict_ensemble(alpha, hparams, model_path=None, diamond='../diamond/',  fasta_path=None, seq_model=None, cut_num='1', taxonid='10090', batch_size=None):
	# predict the GO terms based on the combination of deep learning model and diamond 
	# score = alpha*dl_model + (1-alpha)*diamond
	# argvs:  alpha: the coefficient controling the balance between deep learing model and diamond
	#	  on: 	ontology
	#  	  hparams:  hyperparameters, including the training data path: hparams['train_data'].
	#         model_path: the model path to the deep learning model
	#         diamond: the path to the diamond executable program
	#         fasta_path: the path to the input fasta file.
	#         seq_model:  the list of the input sequences
	#         the above two argcs should only exist one, and  another to be None.
	on = hparams['ontology']
	if fasta_path==None:
		if seq_model==None:
			raise ValueError("Need one inputs from either fasta path or seq model.")
		else:
			print ("seq model input...................for ontolotgy: "+on+"....................")
			fasta_path = './temp.fasta'
			data_to_fasta(seq_model, fasta_path)
			seq_ind={}
			for i in range(len(seq_model)):
				seq_ind[str(i)]=i
	else:
		if seq_model == None:
			print ("fasta sequence input..............for ontolotgy: "+on+".....................")
			seq_model, seq_ind = data_load_fasta(fasta_path)		
		else:
			raise ValueError("Cannot have both fasta and seq model inputs. It will cause ambiguity.")
	seq_tmodel=[]
	ind_tmodel_model={}
	for i in range(len(seq_model)):
		if amino_acid.Nature_seq(seq_model[i]['seq'])==True and len(seq_model[i]['seq'])<=1000:
			seq_tmodel.append(seq_model[i])
			ind_tmodel_model[ len(seq_tmodel)-1] = i

	
	
	pred_dnmd = diamond_pred(seq_ind, fasta_path, hparams['train_data'], diamond, on, cut_num, taxonid)
	print ("pred_dnmd shape:", len(pred_dnmd), len(pred_dnmd[0]))	
	ensemble=np.zeros((len(seq_model), len(pred_dnmd[0])), dtype=np.float32)

	if alpha!=0:
		pred_tmodel = predict_trainedmodel(model_path, seq_tmodel, batch_size=batch_size)
		print ("pred_tmodel:", len(pred_tmodel), len(pred_tmodel[0]))
		
		if 'deepgoplus' in model_path:
			# if the model is deepgoplus, we need to slice the output score matrix.
			print ("#go terms in three ontologies:", hparams['nb_classes_mf'], hparams['nb_classes_bp'], hparams['nb_classes_cc'])
			if hparams['ontology']=='mf':
				pred_tmodel = pred_tmodel[:, 0:hparams['nb_classes_mf']]
			elif hparams['ontology']=='bp':
				pred_tmodel = pred_tmodel[:, hparams['nb_classes_mf']: hparams['nb_classes_mf']+hparams['nb_classes_bp']]
			elif hparams['ontology']=='cc':
				pred_tmodel = pred_tmodel[:,  hparams['nb_classes_mf']+hparams['nb_classes_bp']: ]
			else:
				raise ValueError("no ontology specify")


	ensemble+= np.array(pred_dnmd)

	if alpha!=0:
		for i in range(len(pred_tmodel)):
			ind_ens = ind_tmodel_model[i]
			ensemble[ind_ens]+= alpha*np.array(pred_tmodel[i]) - alpha*np.array(pred_dnmd[ind_ens])
	

	return ensemble, seq_model


def predict_trainedmodel(model_path, seq_model, batch_size):
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


	hparams['train']=False

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




def write_results(predict_model_all, seq_model, ontology_list, outputpath ):

	
	with open(outputpath, "w") as f:
		for i in range(len(predict_model_all)):
			sorted_index = np.argsort(-predict_model_all[i])
			for j in range(len(sorted_index)):
					
					ind = sorted_index[j]
					if j==1000 or predict_model_all[i][ind]<0.4 :
						break

					f.write("%s    %s    %.3f\n" %(seq_model[i]['ID'], ontology_list[ind], \
					predict_model_all[i][ind]))







if __name__=='__main__':
	main()
