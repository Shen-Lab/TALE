import pickle
import sys
import numpy as np
with open(sys.argv[1], "rb") as f:
	label = pickle.load(f)

with open(sys.argv[2], "rb") as f:
	seq=pickle.load(f)

for i in range(len(seq)):
	print (len(seq[i]['seq']), len(seq[i]['GO']), len(label[i]) )
	t1=len(np.unique(label[i]))
	assert t1 == len(label[i])

with open("mf_go_1.pickle", "rb") as f:
	mf = pickle.load(f)

with open("bp_go_1.pickle", "rb") as f:
	bp = pickle.load(f)

with open("cc_go_1.pickle", "rb") as f:
	cc = pickle.load(f)


print (len(label), len(mf), len(bp), len(cc), len(seq))
exit(0)
n1=0
n2=0
n3=0
for i in range(len(label)):
	if 0 not in label[i] and len(mf) not in label[i] and len(mf)+len(bp) not in label[i]:
		raise ValueError("jeeee")
	if 0 in label[i]:
		n1+=1
	if len(mf) in label[i]:
		n2+=1
	if len(mf)+len(bp) in label[i]:
		n3+=1

print (n1, n2, n3)

		
