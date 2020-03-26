import tensorflow as tf
import numpy as np
import train
import sys
import os
import pickle
sys.path.insert(0, 'Utils/')
import hparam
import amino_acid
import metric
import predict

def data_load(test_data_seq, test_data_label):
	with open(test_data_seq, "rb") as f:
		test_seq = pickle.load(f)

	with open(test_data_label, "rb") as f:
		test_label = pickle.load(f)

	return test_seq, test_label


if __name__ == '__main__':


	flags = tf.flags
	FLAGS = flags.FLAGS
	flags.DEFINE_string("model_path", None, "path to the model1")
	flags.DEFINE_string("model_path1", 'None', "path to the model2")
	flags.DEFINE_string("model_path2", 'None', "path to the model3")
	flags.DEFINE_string("test_data_seq", None, "path to the test data")
	flags.DEFINE_string("test_data_label", None, "path to the test data")
	flags.DEFINE_string("train_data", "../data/Gene_Ontology/EXP_Swiss_Prot/", "path to the training data and ontology data")
	flags.DEFINE_string("outputpath", "./Results/" , "e")
	#flags.DEFINE_integer("batch_size", 1, "batch size used for evaluating")
	flags.DEFINE_string("ic_path", "../data/Gene_Ontology/EXP_Swiss_Prot/", "path to the inforamtion content")
	flags.DEFINE_float("alpha", 1, "e")
	flags.DEFINE_float("batch_size", None, "")
	flags.DEFINE_string('ontology', None, "")
	flags.DEFINE_string('runseed','10090' ,"If you want to run this program simutaneously for more than one time, you need to give each individual run different runseed. This is due to diamond intermediate file.")

	if FLAGS.model_path==None:
		raise ValueError("Must specify a model to evaluate.")


	with open(FLAGS.model_path+".hparam", "rb") as f:
		hparams=pickle.load(f)

	hparams['train']=False
	hparams['ic_path']=FLAGS.ic_path
	hparams['train_data']=FLAGS.train_data

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
	
	print ("start evaluating model: "+FLAGS.model_path)
	print ("ontology:"+hparams['ontology'])
	print ("test data seq:"+FLAGS.test_data_seq)

	def sparse_to_dense(y,  length):
		out=np.zeros((len(y), length), dtype=np.int32)
		for i in range(len(y)):
			#print (y[i])
			for j in y[i]:
				out[i][j]=1
		#exit(0)
		return out

	test_seq, test_y_sparse = data_load(FLAGS.test_data_seq, FLAGS.test_data_label)
	
	test_y  = sparse_to_dense(test_y_sparse ,hparams['nb_classes_'+hparams['ontology']])

	all_pred_scores=[]

	pred_scores,_ = predict.predict_ensemble(FLAGS.alpha, hparams, \
		model_path=FLAGS.model_path, seq_model=test_seq, batch_size=FLAGS.batch_size, taxonid=FLAGS.runseed)

	all_pred_scores.append(pred_scores)

	if FLAGS.model_path1!='None':
		pred_scores,_ =predict.predict_ensemble(FLAGS.alpha, hparams,  \
                	model_path=FLAGS.model_path1, seq_model=test_seq, batch_size=FLAGS.batch_size, taxonid=FLAGS.runseed) 	
		all_pred_scores.append(pred_scores)

	if FLAGS.model_path2!='None':
		pred_scores,_ =predict.predict_ensemble(FLAGS.alpha, hparams,  \
			model_path=FLAGS.model_path2, seq_model=test_seq, batch_size=FLAGS.batch_size, taxonid=FLAGS.runseed)
		all_pred_scores.append(pred_scores)	
			
	fmax, smin, auprc = metric.main(test_y, np.mean(all_pred_scores, axis=0), hparams)

	print ("alpha:%.2f fmax: %.3f smin: %.3f auprc: %.3f" %(FLAGS.alpha, fmax, smin, auprc))

	if 'SATE' in FLAGS.model_path:
		np.save(FLAGS.outputpath+"/TALE_"+str(FLAGS.alpha)+"_"+hparams['ontology'], np.mean(all_pred_scores, axis=0))
	elif 'deepgoplus' in FLAGS.model_path:
		np.save(FLAGS.outputpath+"/DeepGO_"+str(FLAGS.alpha)+"_"+hparams['ontology'], np.mean(all_pred_scores, axis=0))

	#np.savetxt(FLAGS.outputpath+"/predicted_scores.txt", pred_scores)
	os.system("rm ./diamond"+FLAGS.runseed+".res")
