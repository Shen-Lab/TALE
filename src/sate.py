import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import (
    Input, Dense, Embedding, Conv1D, Flatten, Concatenate,
    MaxPooling1D, Dropout, RepeatVector, Layer
)

def label_embedding(hparams):

	'''	
	label_matrix = np.load(hparams['data_path']+'/'+hparams['ontology']+"_label_matrix_"+hparams['cut_num']+".npy")
	sparse_matrix=[]
	max_len=0
	for i in range(len(label_matrix)):
		a=[]
		print (i, len(label_matrix))
		for j in range(len(label_matrix)):
			if label_matrix[i][j]==1:
				a.append(j)

		sparse_matrix.append(a)
		if max_len < (len(a)):
			max_len=len(a)

	print ("max_len:", max_len)

	for i in range(len(sparse_matrix)):
		for j in range(len(sparse_matrix[i]), max_len):
			sparse_matrix[i].append(hparams['nb_classes'])
		#print (sparse_matrix[i])
	print (np.array(sparse_matrix).shape)
	print(len(sparse_matrix[0]))
	
	np.save(hparams['data_path']+'/'+hparams['ontology']+"_label_matrix_"+hparams['cut_num']+"_sparse.npy", np.array(sparse_matrix))
	exit(0)
	'''
	sparse_matrix = np.load(hparams['data_path']+'/'+hparams['ontology']+"_label_matrix_"+hparams['cut_num']+"_sparse.npy")
	

	label_matrix_tf = tf.constant(np.array(sparse_matrix))
	zero_tensor = tf.constant(0, shape=(1, hparams['hidden_size']), dtype=tf.float32)

	with tf.variable_scope("label_embedding"):
		# Create and initialize weights. The random normal initializer was chosen
		# randomly, and works well.
		label_embedding_weights = tf.get_variable(
		"weights", [hparams['nb_classes'], hparams['hidden_size']],
		initializer=tf.random_normal_initializer(
		0., hparams['hidden_size'] ** -0.5), trainable=True)
	label_embedding_weights = tf.concat([label_embedding_weights, zero_tensor], axis=0)

	embeddings = tf.gather(label_embedding_weights, label_matrix_tf)
	embeddings *= hparams['hidden_size'] ** 0.5
	embeddings = tf.reduce_sum(embeddings, axis=1)
	print (embeddings)
	return embeddings

def joint_similarity(seq_embed, label_embed, hparams):
	# s_matrix shape [1000, nb_classes]
	label_embed = tf.expand_dims(label_embed, axis=0)
	label_embed = tf.tile(label_embed, [hparams['batch_size'],1,1])
	similarity_matrix = tf.matmul(seq_embed, label_embed, transpose_b=True)
	similarity_matrix = tf.nn.softmax(similarity_matrix, axis=-1)

	w  = Conv1D(filters=hparams['joint_similarity_filter_size'],kernel_size=hparams['joint_similarity_kernel_size'],\
	 strides=1, padding='same', activation='relu')(similarity_matrix)
	w1 = tf.math.reduce_max(w, axis=-1)
	print(similarity_matrix)
	
	w1 = tf.nn.softmax(w1, axis=-1)
	w1 = tf.expand_dims(w1, axis=1)

	w2 = tf.matmul(w1,seq_embed)

	return tf.squeeze(w2, axis=1)


if __name__=='__main__':
	label_embedding()
