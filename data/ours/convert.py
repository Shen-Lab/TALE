import numpy
import pickle
import numpy as np
for on in ['mf','bp','cc']:

	with open("../Swiss_Prot/seq_"+on+'.pkl', "rb") as f:
		seq = pickle.load(f)
	with open("../ontology/"+on+"_go_trunc.pickle", "rb") as f:
		graph= pickle.load(f)


	train_seq=[]
	test_seq=[]
	train_label=[]
	test_label=[]

	label_matrix_1_sparse=[]
	label_regular_1 = []


	for i in seq:
		if seq[i]['mode'] == 'train':
			train_seq.append(seq[i])
			train_label.append(seq[i]['label'])
		else:

			test_seq.append(seq[i])
			test_label.append(seq[i]['label'])			


	maxlen=0
	for go in graph:
		maxlen=max(maxlen, len(graph[go]['label']))
	for go in graph:
		label_matrix_1_sparse.append(graph[go]['label']+ ([len(graph)] * (maxlen-len(graph[go]['label']))))

		for i in graph[go]['child']:
			label_regular_1.append([graph[go]['ind'], graph[i]['ind']])


	label_matrix_1_sparse=np.array(label_matrix_1_sparse)
	label_regular_1  = np.array(label_regular_1)

	q1 = np.load("/home/cyppsp/project_HMC/data/Gene_Ontology/EXP_Swiss_Prot/"+on+"_label_matrix_1.npy")
	q2 = np.load("/home/cyppsp/project_HMC/data/Gene_Ontology/EXP_Swiss_Prot/"+on+"_label_regular_1.npy")

	print (label_matrix_1_sparse.shape, q1.shape, label_regular_1.shape, q2.shape)
	
	with open("train_seq_"+on, "wb") as f:
		pickle.dump(train_seq, f)
	with open("test_seq_"+on, "wb") as f:
		pickle.dump(test_seq, f)

	with open("train_label_"+on, "wb") as f:
		pickle.dump(train_label, f)
	with open("test_label_"+on, "wb") as f:
		pickle.dump(test_label, f)		


	np.save(on+"_label_matrix_1_sparse", label_matrix_1_sparse)
	np.save(on+"_label_regular_1", label_regular_1)