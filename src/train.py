# created 1/9/2020
# author: Yue Cao
# The main training program for TALE.

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


def cut(a, maxlen):
   b=[]
   for i in a:
      if len(i)>maxlen:
          start = np.random.randint(i-maxlen+1)
          b.append(i[start: start+maxlen])
      else:
          b.append(i)
   return b


class HMC_models(object):

	def __init__(self, hparams):
		self.hparams = hparams	

	def Embedding(self, x):

		# args:   x shape: [ batch_size, length]
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

	def Encoder(self, encoder_inputs):
		# args: encoder_inputs: shape:[batch_size, length, hidden_size]
		# return:  [batch_size, length, hidden]
		# Transfomer-based sequence encoder
		hparams=self.hparams
		if hparams['encoding_model'] == 'transformer':
			self.encoder_stack = transformer_encode.EncoderStack(hparams)
			return self.encoder_stack(encoder_inputs, self.attention_bias, self.inputs_padding)

	def Main_model(self):
		hparams=self.hparams
		inputs = tf.placeholder(shape=(self.hparams['batch_size'], self.hparams['MAXLEN']), dtype=tf.int32)
		outs   = tf.placeholder(shape=(self.hparams['batch_size'], self.hparams['nb_classes']), dtype=tf.int32)
		return_box = [inputs, outs]

		if hparams['main_model'] == 'SALT':
			encoder_inputs = self.Embedding(inputs)
			encoder_outputs = self.Encoder(encoder_inputs)

			def output_layer(input):
				# argv the input is a tensor with shape [batch, length, hidden_size]
				# we do average over the second dimensions.
					#out1 = tf.math.reduce_mean(input, 1)
					out1 = tf.keras.layers.MaxPool1D(8, data_format='channels_first')(input)
					out1 = tf.reshape(out1, [hparams['batch_size'], -1])
					print (out1)
					
					out2 = tf.layers.Dense(hparams['nb_classes'], activation='sigmoid', name='dense_out')(out1)
					return out2

			if (hparams['label_embed']==False):

				pred_out = output_layer(encoder_outputs)
				loss = self.loss(outs, pred_out, 'bc')

				return_box.append(loss)
				return_box.append(pred_out)
			else:
				label_embed= sate.label_embedding(hparams)
				out1 = sate.joint_similarity(encoder_outputs,label_embed,hparams)
				pred_out=tf.layers.Dense(hparams['nb_classes'], activation='sigmoid', name='dense_out')(out1)
				print ('pred_out', pred_out.shape)
				loss = self.loss(outs, pred_out, 'bc')
				loss = loss + hparams['regular_lambda']*self.regular_loss(pred_out)

				return_box.append(loss)
				return_box.append(pred_out)
				return_box.append(self.regular_loss(pred_out))



		else:
			raise ValueError("couldnt find the main model.")
		if len (return_box)<5:
			return_box.append(tf.constant([0]))
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

	def train(self):

		hparams=self.hparams
		data = self.data_load(self.hparams['data_path'])

		def sparse_to_dense(y,  length):
			out=np.zeros((len(y), length), dtype=np.int32)
			for i in range(len(y)):
				#print (y[i])
				for j in y[i]:
					out[i][j]=1
			#exit(0)
			return out
			
		with tf.device('/gpu:0'):
			holder_list = self.Main_model()  #------------holder_list: [model_input, model_output, loss]
			# optimization
			optimizer = tf.train.AdamOptimizer(learning_rate=self.hparams['lr'])
			train_op = optimizer.minimize(holder_list[2])
			init_op = tf.global_variables_initializer()
		print ("#trainable_variables:", np.sum([np.prod(v.shape) for v in tf.trainable_variables()]))
		batch_size = self.hparams['batch_size']
		
		train_x = data[0]
		train_y  = sparse_to_dense(data[1] ,hparams['nb_classes'])
		val_x = data[2]
		val_y= sparse_to_dense(data[3], hparams['nb_classes'])
		val_list=[v for v in tf.global_variables()]
		saver = tf.train.Saver(val_list, max_to_keep=None)
		
		print ('start training. training information:')
		if hparams['resume_model']!=None:
			print ('resume model?: Yes and'+hparams['resume_model'])
		else:
			print ('resume model?: No, training untrained model')
		for i in hparams:
			print (i,' : ', hparams[i])	

		with tf.Session() as sess:
			if hparams['resume_model']!=None:
				saver.restore(sess, hparams['resume_model'])
				resume_epoch = int(hparams['resume_model'].split('_')[-1])
			else:
				sess.run(init_op)
				resume_epoch = 0
			epoch_train_loss=[]
			epoch_val_loss=[]				

			for epoch in range(hparams['epochs']):

				sepoch_train_loss = 0.

				iterations = int((len(train_x)) // hparams['batch_size'])
				print ("epoch %d begins:" %(resume_epoch+epoch+1))
				print ("#iterations:", iterations)
				for ite in range(iterations):
					x = cut(train_x[ite*batch_size: (ite+1)*batch_size], hparams['MAXLEN'])
					y = train_y[ite*batch_size: (ite+1)*batch_size]

					train_loss ,_ , regular_loss= sess.run([holder_list[2], train_op, holder_list[4]], {holder_list[0]: x, holder_list[1]: y})

					sepoch_train_loss+=train_loss
					print ("iteration %d/%d totaltrain_loss: %.3f regular loss %.3f" %(ite+1, iterations, train_loss, regular_loss))

				sepoch_train_loss /= iterations
				with open(hparams['save_path']+"/train_log", "a") as f:
					f.write("%d %.6f\n" %(epoch+resume_epoch+1, sepoch_train_loss))

				# shuffle training data
				train_z = list(zip(train_x, train_y))
				shuffle(train_z)
				train_x, train_y = zip(*train_z)

			
			#  run evaluation..............
			epoch = hparams['epochs']-1
			pred_scores=[]
			sepoch_val_loss = 0.
			iterations = int(len(val_x) // hparams['batch_size'])
			for ite in range(iterations):
				x= val_x[ite*batch_size: (ite+1)*batch_size]
				y= val_y[ite*batch_size: (ite+1)*batch_size]
				val_loss, pred_score = sess.run([holder_list[2], holder_list[3]], {holder_list[0]: x, holder_list[1]: y})
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
		train_Y= train_label
		val_X=[]
		val_Y= val_label

		for i in range(len(train_seq)):
			train_X.append(amino_acid.to_int(train_seq[i]['seq'], self.hparams))
		
		for i in range(len(val_seq)):
			val_X.append(amino_acid.to_int(val_seq[i]['seq'], self.hparams))

		return train_X, train_Y, val_X, val_Y

if __name__== "__main__":


	flags = tf.flags
        #--------------------------------------------------training HParams:
	flags.DEFINE_string("main_model", "SALT", "e")
	flags.DEFINE_integer("batch_size", 32, "e")
	flags.DEFINE_integer("epochs", 100, "e")
	flags.DEFINE_float("lr", 1e-3, "e")
	flags.DEFINE_string("save_path", './', "model savepath")
	flags.DEFINE_string("resume_model", None, "")
	flags.DEFINE_string("ontology", 'mf', "e")
	flags.DEFINE_integer("nb_classes", None, "e")
	flags.DEFINE_bool("label_embed", True, "e")
	flags.DEFINE_string("data_path", '../data/ours/', "path_to_store_data")
	flags.DEFINE_float("regular_lambda", 0, "e")
	flags.DEFINE_string("cut_num", '1', "e")
	flags.DEFINE_float("l2_lambda", 0, "e")
	flags.DEFINE_integer("num_heads", 2, "e")
	flags.DEFINE_integer("num_hidden_layers", 6, "e")
	flags.DEFINE_integer("hidden_size", 64, "e")
	FLAGS = flags.FLAGS


	hparams = hparam.params(flags)

	#  if we resume a model, we need to read the hyperparameter file.
	if hparams['resume_model']!=None:
		resume_model = hparams['resume_model']
		with open(hparams['resume_model']+'.hparam', "rb") as f:
			hparams=pickle.load(f)
		hparams['resume_model']=resume_model

	model1 = HMC_models(hparams)
	model1.train()

