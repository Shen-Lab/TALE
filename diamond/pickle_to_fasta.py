import pickle
import sys

input_path = sys.argv[1]
output_path = sys.argv[2]

with open(input_path, "rb") as f:
	seq_sp = pickle.load(f)

with open(output_path, "w") as f:
	for i in range(len(seq_sp)):
		f.write('>'+str(i)+'\n')
		f.write(seq_sp[i]['seq'])
		f.write('\n')

