# created 1/11/2020
# author: Yue Cao

import tensorflow as tf
import pickle
import sys
def params(flags):

	hparams={}
	FLAGS = flags.FLAGS

	hparams['MAXLEN'] = 1000

	#--------------------------------------------------training HParams:
	hparams['main_model'] = FLAGS.main_model	

	hparams['batch_size'] = FLAGS.batch_size

	hparams['epochs'] = FLAGS.epochs

	hparams['lr'] = FLAGS.lr
	
	hparams['save_path'] = FLAGS.save_path

	
	hparams['resume_model'] = FLAGS.resume_model

	#------------------------------------------------Protein Function Predition HParams:
	#---------------------------------final classes:
	#-----------MFO: 2232
	#-----------BPO: 6997
	#-----------CCO: 1070
	hparams['ontology'] = FLAGS.ontology
	hparams['data_path'] = FLAGS.data_path
	

	hparams['cut_num']=FLAGS.cut_num
	hparams['tsnet_cutnum'] = '1'

	if 'EXP' in hparams['data_path'] and 'SALT' not in hparams['main_model']:
		with open(hparams['data_path']+'/mf_go_'+hparams['cut_num']+".pickle", "rb") as f:
                        go_dag1 = pickle.load(f)
		with open(hparams['data_path']+'/bp_go_'+hparams['cut_num']+".pickle", "rb") as f:
			go_dag2 = pickle.load(f)
		with open(hparams['data_path']+'/cc_go_'+hparams['cut_num']+".pickle", "rb") as f:
			go_dag3 = pickle.load(f)
		hparams['nb_classes']= len(go_dag1)+len(go_dag2)+len(go_dag3)
		hparams['nb_classes_mf']=len(go_dag1)
		hparams['nb_classes_bp']=len(go_dag2)
		hparams['nb_classes_cc']=len(go_dag3)
	else:	
		with open(hparams['data_path']+'/'+hparams['ontology']+"_go_"+hparams['cut_num']+".pickle", "rb") as f:
			go_dag = pickle.load(f)
		hparams['nb_classes']=len(go_dag)
		

	flags.DEFINE_integer("vocab_size", 26, "e")
	hparams['vocab_size'] = FLAGS.vocab_size


	#---------------------------------------------------Transformer HParams:
	hparams['hidden_size'] = FLAGS.hidden_size

	hparams['num_hidden_layers'] = FLAGS.num_hidden_layers
	hparams['num_heads'] = FLAGS.num_heads
	flags.DEFINE_bool("train", True, "e")
	hparams['train'] = FLAGS.train
	
	#------------------------------------------------------SATE HParams:
	hparams['joint_similarity_kernel_size'] = 10
	hparams['joint_similarity_filter_size'] = 10
	hparams['regular_lambda'] = FLAGS.regular_lambda


	#ts-net HParams:
	hparams['tsnet_filter'] = [10, 5]
	hparams['tsnet_kernel'] = [10, 5]
	hparams['tsnet_stride'] = [5, 2]
	hparams['tsnet_pool'] = [5, 2] 
	hparams['l2_lambda'] = FLAGS.l2_lambda
	#----------------------------------------------------------model modules
	flags.DEFINE_string("embedding_model", "transformer", "e")
	hparams['embedding_model'] = FLAGS.embedding_model

	flags.DEFINE_string("encoding_model", "transformer", "e")
	hparams['encoding_model'] = FLAGS.encoding_model

	hparams['label_embed'] = FLAGS.label_embed




	hparams['layer_postprocess_dropout']=0.1
	hparams['attention_dropout']=0.1
	hparams['relu_dropout']=0.1
	hparams['filter_size'] =64



	# TPU 
	hparams['use_tpu']  = False
	hparams['allow_ffn_pad']=True


	return  hparams

if __name__== "__main__":


	flags = tf.flags
	#--------------------------------------------------training HParams:
	flags.DEFINE_string("main_model", "SALT", "e")
	flags.DEFINE_integer("batch_size", 256, "e")
	flags.DEFINE_integer("epochs", 20, "e")
	flags.DEFINE_float("lr", 1e-5, "e")
	flags.DEFINE_string("save_path", '../trained_models/', "model savepath")
	flags.DEFINE_string("resume_model", None, "")
	flags.DEFINE_string("ontology", 'cc', "e")
	flags.DEFINE_integer("nb_classes", 1070, "e")
	flags.DEFINE_bool("label_embed", False, "e")
	hparams = params(flags)

	with open (sys.argv[1]+".hparam", "wb") as f:	
		pickle.dump(hparams, f)	
