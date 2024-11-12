import numpy as np
import sys
import pickle
from random import shuffle
# add encode path
sys.path.insert(0, './Protein_Encode/')
import attention_layer
import ffn_layer
import embedding_layer
import model_utils
import transformer_encode
import sate

sys.path.insert(0, 'Utils/')
import hparam
import amino_acid
import metric

import tensorflow as tf
flags = tf.flags
logging = tf.logging

from proteinMPNN_utils import ProteinMPNN_Encoder
import torch

def cut(a, maxlen):
   b=[]
   for i in a:
      if len(i)>maxlen:
          start = np.random.randint(0, len(i)-maxlen+1)
          b.append(i[start: start+maxlen])
      else:
          b.append(i)
   return b

def all_cut(seq_batch, coor_batch, length, maxlen, with_stru = True):
    """
    prepare the inputs for ProteinMPNN; by SZ
    Args:
        seq: bz x max(maxlen, L)
        coor: bz x max(maxlen, L) x 4 x 3
        length: bz
    """
    seq_new = []
    coor_new = []
    mask_batch = []
    residue_idx = []
    chain_encoding_all = []

    for i,seq in enumerate(seq_batch):
        coor = coor_batch[i]
        seq_len = len(seq)

        if with_stru and (seq_len != coor.shape[0]):
            print('Warining! Sequence length (%d) and the structure shape (%d) do not match!'%(seq_len, coor.shape[0]))
            continue

        if seq_len > maxlen:
            start = np.random.randint(0, seq_len - maxlen+1)
            seq_new.append(seq[start: start+maxlen])
            coor_new.append(coor[start: start+maxlen])
            mask_batch.append(np.ones(maxlen, dtype=np.int32))
            residue_idx.append(np.arange(maxlen)) 
            chain_encoding_all.append(np.ones(maxlen, dtype=np.int32))
        else:
            l = length[i]
            seq_new.append(seq)
            coor_new.append(coor)
            mask = np.zeros(maxlen, dtype=np.int32)
            mask[:l] = 1
            mask_batch.append(mask)
            r_idx = -100*np.ones(maxlen, dtype=np.int32)
            r_idx[:l] = np.arange(l)
            residue_idx.append(r_idx)
            chain_idx = np.zeros(maxlen, dtype=np.int32)
            chain_idx[:l] = 1
            chain_encoding_all.append(chain_idx) 
    
    return seq_new, coor_new, mask_batch, residue_idx, chain_encoding_all


class HMC_models(object):

	def __init__(self, hparams):
		self.hparams = hparams

		### structure information
		if hparams['with_structure']:
			self.structure_encode = ProteinMPNN_Encoder(node_features = hparams['struc_dim'],
								edge_features = hparams['struc_dim'],
								hidden_dim = hparams['struc_dim'],
								num_encoder_layers = hparams['struc_nlayers'],
								k_neighbors = hparams['struc_kneighbor'],
								dropout = hparams['struc_dropout'],
								augment_eps = hparams['struc_eps']).cuda()

			self.strucure_optimizer = torch.optim.Adam(self.structure_encode.parameters(), lr = hparams['lr'])

	################################### Sequence Embedding ###################################
	def Embedding(self, x):
		## Sequence embedding
		# args:   x: sequence_feature, [ batch_size, length]
		# return: [batch_size, length, hidden_size]

		hparams=self.hparams
		if hparams['embedding_model']=='transformer':

			self.embedding_layer = embedding_layer.EmbeddingSharedWeights(
				hparams["vocab_size"], hparams["hidden_size"])

			embedded_inputs = self.embedding_layer(x)
			with tf.name_scope("add_pos_encoding"):
				length = tf.shape(embedded_inputs)[1]
				pos_encoding = model_utils.get_position_encoding(
			    	length, hparams["hidden_size"])
				encoder_inputs = embedded_inputs + pos_encoding


			if self.hparams['train']:
					encoder_inputs = tf.nn.dropout(
						encoder_inputs, rate=self.hparams["layer_postprocess_dropout"])
			
			self.inputs_padding = model_utils.get_padding(x)
			self.attention_bias = model_utils.get_padding_bias(x)
			return encoder_inputs

        ################################### Feature Encoding ###################################
	def Encoder(self, encoder_inputs):
		# args: encoder_inputs: shape:[batch_size, length, hidden_size]
		# return:  [batch_size, length, hidden]
		# Transfomer-based sequence encoder
		hparams=self.hparams
		if hparams['encoding_model'] == 'transformer':
			self.encoder_stack = transformer_encode.EncoderStack(hparams)
			return self.encoder_stack(encoder_inputs, self.attention_bias, self.inputs_padding)

        ################################### Model ###################################
	def Main_model(self):
		hparams=self.hparams
		inputs = tf.placeholder(shape=(self.hparams['batch_size'], self.hparams['MAXLEN']), dtype=tf.int32)

		if hparams['with_structure']:
			strut_emb = tf.placeholder(shape=(self.hparams['batch_size'], self.hparams['MAXLEN'], self.hparams['struc_dim']), dtype=tf.float32)
			outs   = tf.placeholder(shape=(self.hparams['batch_size'], self.hparams['nb_classes']), dtype=tf.int32)
		else:
			strut_emb = tf.constant([0])
			outs   = tf.placeholder(shape=(self.hparams['batch_size'], self.hparams['nb_classes']), dtype=tf.int32)

		return_box = [inputs, strut_emb, outs]

		if hparams['main_model'] == 'SALT':
			encoder_inputs = self.Embedding(inputs)
			encoder_outputs = self.Encoder(encoder_inputs)
			# print(encoder_outputs.shape, strut_emb.shape)

			###### concat the sequence embedding and the structure embedding ######
			if hparams['with_structure']: # by SZ	
				encoder_outputs = tf.concat(values = [encoder_outputs, strut_emb], axis = -1)

				if hparams['label_embed']:
					def stru_seq(input): # to concatenate the seq_emb and structure_emb, and get the desired dim
                                        	out = tf.layers.Dense(hparams['hidden_size'], activation=None, name='seq_stru')(input)
                                        	return out
					encoder_outputs = stru_seq(encoder_outputs)  # SZ

			# print(encoder_outputs.shape, encoder_outputs.dtype)

			########################################################################

			def output_layer(input):
				# argv the input is a tensor with shape [batch, length, hidden_size]
				# we do average over the second dimensions.
					#out1 = tf.math.reduce_mean(input, 1)
					out1 = tf.keras.layers.MaxPool1D(8, data_format='channels_first')(input)
					out1 = tf.reshape(out1, [hparams['batch_size'], -1])
					print (out1)
					
					out2 = tf.layers.Dense(hparams['nb_classes'], activation='sigmoid', name='dense_out')(out1)
					return out2

			if (not hparams['label_embed']):
				print('Without joint embedding.')
				pred_out = output_layer(encoder_outputs)
				loss = self.loss(outs, pred_out, 'bc')
				regu_loss = tf.constant([0])

			else:
				print('With joint embedding.')
				label_embed= sate.label_embedding(hparams)
				out1 = sate.joint_similarity(encoder_outputs,label_embed,hparams)
				pred_out=tf.layers.Dense(hparams['nb_classes'], activation='sigmoid', name='dense_out')(out1)
				#print ('pred_out', pred_out.shape)
				
				loss = self.loss(outs, pred_out, 'bc')
				regu_loss = self.regular_loss(pred_out)
				loss = loss + hparams['regular_lambda'] * regu_loss

			if hparams['with_structure']:
				struc_op = tf.gradients(loss, strut_emb)
			else:
				struc_op = tf.constant([0])

			return_box.append(loss)
			return_box.append(pred_out)
			return_box.append(regu_loss)
			return_box.append(struc_op)

		else:
			raise ValueError("couldnt find the main model.")

		return return_box

	def loss(self, ytrue, ypred, loss_type):
		if loss_type == 'bc':
			bce=tf.keras.losses.BinaryCrossentropy()
			return bce(ytrue, ypred)

	def regular_loss(self, pred_out):
		hparams=self.hparams
		label_regular = np.load(hparams['data_path']+'/'+hparams['ontology']+"_label_regular_"+hparams['cut_num']+".npy")

		ind_fa = tf.constant(label_regular.transpose()[0])
		ind_child = tf.constant(label_regular.transpose()[1])

		print (ind_fa.shape, pred_out.shape)		
		r1_fa = tf.gather(pred_out, ind_fa,axis=1)
		r1_child = tf.gather(pred_out, ind_child, axis=1)
		
		regular_loss =tf.nn.relu( tf.math.subtract(r1_child, r1_fa) )
		regular_loss = tf.reduce_mean(regular_loss)
		print ("regular_loss", regular_loss)
		return regular_loss

        ################################### Training Pipeline ###################################
	def train(self):
		hparams=self.hparams

		def sparse_to_dense(y,  length):
			out=np.zeros((len(y), length), dtype=np.int32)
			for i in range(len(y)):
				#print (y[i])
				for j in y[i]:
					out[i][j]=1
			#exit(0)
			return out

		############ training setting ############
		with tf.device('/gpu:0'):
			holder_list = self.Main_model()  #------------holder_list: [model_input, model_output, loss]
			# optimization
			optimizer = tf.train.AdamOptimizer(learning_rate=self.hparams['lr'])
			train_op = optimizer.minimize(holder_list[3])
			init_op = tf.global_variables_initializer()
		print ("#trainable_variables:", np.sum([np.prod(v.shape) for v in tf.trainable_variables()]))

		val_list=[v for v in tf.global_variables()]
		saver = tf.train.Saver(val_list, max_to_keep=None)

		############ data loading ############
		data = self.data_load(self.hparams['data_path']) # train_X, train_coor, train_len, train_Y, val_X, val_coor, val_len, val_Y
		batch_size = self.hparams['batch_size']
		
		train_x = data[0]
		train_coor = data[1]
		train_len = data[2]
		train_y = sparse_to_dense(data[3] ,hparams['nb_classes'])

		val_x = data[4]
		val_coor = data[5]
		val_len = data[6]
		val_y = sparse_to_dense(data[7], hparams['nb_classes'])

                ############ Training ############
		print ('start training. training information:')
		if hparams['resume_model']!=None:
			print ('resume model?: Yes and'+hparams['resume_model'])
		else:
			print ('resume model?: No, training untrained model')
		for i in hparams:
			print (i,' : ', hparams[i])	

		###### run session ######
		with tf.Session() as sess:
			if hparams['resume_model']!=None:
				saver.restore(sess, hparams['resume_model'])
				resume_epoch = int(hparams['resume_model'].split('_')[-1])
			else:
				sess.run(init_op)
				resume_epoch = 0
			epoch_train_loss=[]
			epoch_val_loss=[]				

			### epoch wise
			for epoch in range(hparams['epochs']):

				sepoch_train_loss = 0.
				iterations = int((len(train_x)) // hparams['batch_size'])
				print ("epoch %d begins:" %(resume_epoch+epoch+1))
				print ("#iterations:", iterations)

				### iteration wise
				for ite in range(iterations):
					# x = cut(train_x[ite*batch_size: (ite+1)*batch_size], hparams['MAXLEN']) # sequence
					x = train_x[ite*batch_size: (ite+1)*batch_size] # sequence (by SZ)
					y = train_y[ite*batch_size: (ite+1)*batch_size] # label

					coor = train_coor[ite*batch_size: (ite+1)*batch_size] # structure
					length = train_len[ite*batch_size: (ite+1)*batch_size]

					x, coor, mask, residue_idx, chain_encoding_all = all_cut(x, coor, length, hparams['MAXLEN'])
					# x: bz x L
					# coor: bz x L x 4 x 3
					# mask: bz x L
					# residue_idx: bz x L
					# chain_encoding_all: bz x L
					if len(x) == 0:
						print('Warining! Empty batch found!')
						continue

					if hparams['with_structure']:
						# print(type(coor))
						coor = np.array(coor)
						print('coor:', coor.shape)
						if len(coor.shape) < 2:
							print('Warining! Abnormal coordinates found!', coor)
							continue

						struc_emb = self.structure_encode(
										X = torch.from_numpy(coor).cuda(), 
										mask = torch.from_numpy(np.array(mask, dtype=np.int32)).cuda(), 
										residue_idx = torch.from_numpy(np.array(residue_idx, dtype=np.int32)).cuda(), 
										chain_encoding_all = torch.from_numpy(np.array(chain_encoding_all, dtype=np.int32)).cuda())
						#print('Structure embedding', struc_emb.shape)
						struc_emb_numpy = struc_emb.detach().cpu().numpy()
						#print('Structure embedding', struc_emb.shape, type(struc_emb), type(struc_emb_numpy))
						#print(struc_emb_numpy[0,0])
						#quit()

						train_loss, regular_loss, train_grad = sess.run([holder_list[3], holder_list[5], holder_list[6]], {holder_list[0]: x, holder_list[1]: struc_emb_numpy, holder_list[2]: y})
						print('Loss:', train_loss)
						# quit()
						# for ProteinMPNN
						# print(torch.tensor(train_grad).shape)
						struc_emb.backward(torch.tensor(train_grad).squeeze(0).cuda())
						self.strucure_optimizer.step()
						# self.strucure_optimizer.backward(torch.tensor(train_grad).squeeze(0).cuda())
					else:
						train_loss, regular_loss = sess.run([holder_list[3], holder_list[5]], {holder_list[0]: x, holder_list[2]: y}) 

					sepoch_train_loss+=train_loss
					#print(ite+1, iterations, train_loss, regular_loss)
					print("iteration %d/%d totaltrain_loss: %.3f regular loss %.3f" %(ite+1, iterations, train_loss, regular_loss))

				sepoch_train_loss /= iterations
				with open(hparams['save_path']+"/train_log", "a") as f:
					f.write("%d %.6f\n" %(epoch+resume_epoch+1, sepoch_train_loss))

				# shuffle training data
				train_z = list(zip(train_x, train_y))
				shuffle(train_z)
				train_x, train_y = zip(*train_z)
				
				if((epoch)%1==0):
					saver.save(sess, hparams['save_path']+"/ckpt_"+"epoch_"+str(epoch+resume_epoch+1))
					with open(hparams['save_path']+"/ckpt_"+"epoch_"+str(epoch+resume_epoch+1)+".hparam","wb" ) as f:
						pickle.dump(hparams,f)

			
			#  run evaluation..............
			epoch = hparams['epochs']-1
			pred_scores=[]
			sepoch_val_loss = 0.
			iterations = int(len(val_x) // hparams['batch_size'])
			for ite in range(iterations):
				x= val_x[ite*batch_size: (ite+1)*batch_size]
				y= val_y[ite*batch_size: (ite+1)*batch_size]

				coor = val_coor[ite*batch_size: (ite+1)*batch_size] # structure
				length = val_len[ite*batch_size: (ite+1)*batch_size]

				x, coor, mask, residue_idx, chain_encoding_all = all_cut(x, coor, length, x.shape[1])

				if hparams['with_structure']:
					struc_emb = self.structure_encode(
									X = torch.from_numpy(np.array(coor)).cuda(), 
									mask = torch.from_numpy(np.array(mask, dtype=np.int32)).cuda(),  
									residue_idx = torch.from_numpy(np.array(residue_idx, dtype=np.int32)).cuda(), 
									chain_encoding_all = torch.from_numpy(np.array(chain_encoding_all, dtype=np.int32)).cuda())
								
					val_loss, pred_score = sess.run([holder_list[3], holder_list[4]], {holder_list[0]: x, holder_list[1]: struc_emb.detach().cpu().numpy(), holder_list[2]: y})
				else:
					val_loss, pred_score = sess.run([holder_list[3], holder_list[4]], {holder_list[0]: x, holder_list[2]: y})

				sepoch_val_loss+=val_loss
				pred_scores.extend(pred_score)
				print ("iteration %d/%d val_loss: %.3f" %(ite+1, iterations, val_loss))

			sepoch_val_loss/=iterations

			print ("epoch %d, train_loss: %.3f val_loss: %.3f" %(epoch+resume_epoch+1, sepoch_train_loss, sepoch_val_loss))

			
			fmax, smin, auprc = metric.main(val_y[:len(pred_scores)], pred_scores, hparams)
			with open(hparams['save_path']+"/val_log", "a") as f:
				f.write("%d %.3f %.3f %.3f %.3f\n" %(epoch+resume_epoch+1, sepoch_val_loss, fmax, smin, auprc))		
			print ("epoch %d %.3f %.3f %.3f %.3f\n" %(epoch+resume_epoch+1, sepoch_val_loss, fmax, smin, auprc))		
			
			# save_model..................
			saver.save(sess, hparams['save_path']+"/ckpt_"+"epoch_"+ \
				str(epoch+resume_epoch+1))
			with open(hparams['save_path']+"/ckpt_"+"epoch_"+ \
				str(epoch+resume_epoch+1)+".hparam","wb" ) as f:
				pickle.dump(hparams,f)


	def data_load(self, path):

		with open(path+"/train_seq_"+hparams['ontology'], "rb") as f:
			train_seq = pickle.load(f)

		with open(path+"/train_label_"+hparams['ontology'], "rb") as f:
			train_label = pickle.load(f)

		with open(path+"/test_seq_"+hparams['ontology'], "rb") as f:
			val_seq = pickle.load(f)

		with open(path+"/test_label_"+hparams['ontology'], "rb") as f:
			val_label = pickle.load(f)

		train_X=[]
		train_coor = []
		train_len = []
		train_Y= train_label

		val_X=[]
		val_coor = []
		val_len = []
		val_Y= val_label

		for i in range(len(train_seq)):
			# print(train_seq[i].keys())
			train_X.append(amino_acid.to_int(train_seq[i]['seq'], self.hparams)) # entry shape: (L_max,)
			train_coor.append(amino_acid.coor_load(train_seq[i]['structure'], self.hparams)) # entry shape: (L_max, 4, 3)
			l = len(train_seq[i]['seq'])
			train_len.append(l)
		
		for i in range(len(val_seq)):
			val_X.append(amino_acid.to_int(val_seq[i]['seq'], self.hparams)) # entry shape: (L_max,)val_coor.append(amino_acid.coor_load(val_seq[i]['structure'], self.hparams)) # entry shape: (L_max, 4, 3)
			val_coor.append(amino_acid.coor_load(val_seq[i]['structure'], self.hparams)) # entry shape: (L_max, 4, 3)
			l = len(val_seq[i]['seq'])
			val_len.append(l)

		return train_X, train_coor, train_len, train_Y, val_X, val_coor, val_len, val_Y

if __name__== "__main__":


	flags = tf.flags
        #--------------------------------------------------training HParams:
	flags.DEFINE_string("main_model", "SALT", "e")
	flags.DEFINE_integer("batch_size", 32, "e")
	flags.DEFINE_integer("epochs", 100, "e")
	flags.DEFINE_float("lr", 1e-3, "e")
	flags.DEFINE_string("save_path", '../logs_debug/', "model savepath")
	flags.DEFINE_string("resume_model", None, "")
	flags.DEFINE_string("ontology", 'mf', "e")
	flags.DEFINE_integer("nb_classes", None, "e")
	flags.DEFINE_integer("label_embed", 1, "e")
	flags.DEFINE_string("data_path", '../data/TALE_2/', "path_to_store_data")
	flags.DEFINE_float("regular_lambda", 0, "e")
	flags.DEFINE_string("cut_num", '1', "e")
	flags.DEFINE_float("l2_lambda", 0, "e")
	flags.DEFINE_integer("num_heads", 2, "e")
	flags.DEFINE_integer("num_hidden_layers", 6, "e")
	flags.DEFINE_integer("hidden_size", 64, "e")
	### for ProteinMPNN
	flags.DEFINE_integer("with_structure", 1, "whether incorporate the structure info") 
	flags.DEFINE_integer("struc_dim", 64, "hidden dim of ProteinMPNN")
	flags.DEFINE_integer("struc_nlayers", 3, "layer amount of ProteinMPNN")
	flags.DEFINE_integer("struc_kneighbor", 32, "k of kNN of ProteinMPNN")
	flags.DEFINE_float("struc_dropout", 0.1, "dropout rate of ProteinMPNN")
	flags.DEFINE_float("struc_eps", 0.1, "augment_eps of ProteinMPNN")

	flags.FLAGS.label_embed = bool(flags.FLAGS.label_embed)
	FLAGS = flags.FLAGS

	hparams = hparam.params(flags)

	#  if we resume a model, we need to read the hyperparameter file.
	if hparams['resume_model']!=None:
		resume_model = hparams['resume_model']
		with open(hparams['resume_model']+'.hparam', "rb") as f:
			hparams=pickle.load(f)
		hparams['resume_model']=resume_model

	hparams['with_structure'] = bool(FLAGS.with_structure)
	hparams['struc_dim'] = FLAGS.struc_dim
	hparams['struc_nlayers'] = FLAGS.struc_nlayers
	hparams['struc_kneighbor'] = FLAGS.struc_kneighbor
	hparams['struc_dropout'] = FLAGS.struc_dropout
	hparams['struc_eps'] = FLAGS.struc_eps

	if hparams['with_structure']:
		print('Incorporating the structure information.')
	else:
		print('Only take the sequences as the inputs.')

	model1 = HMC_models(hparams)
	model1.train()

