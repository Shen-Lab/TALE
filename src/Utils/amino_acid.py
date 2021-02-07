amino_acid = ['A', 'R', 'N', 'D',  'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
import numpy as np
AA_indx={'pad': 0, 'A': 1, 'R':2, 'N':3, 'D':4,  'C':5, 'E':6, 'Q':7, 'G':8, 'H':9, 'I':10,\
 'L':11, 'K':12, 'M':13, 'F':14, 'P':15, 'S':16, 'T':17, 'W':18, 'Y':19, 'V':20, 'B':21, 'O':22, 'U':23, 'X':24, 'Z':25}
    
def Nature_seq(seq):

	for i in seq:
		if i not in amino_acid:
			return False

	return True



import matplotlib.pyplot as plt
def seq_length_plot(seq,name='seq_dis', maxlen=2000):

        length = []
        for i in seq:
                length.append(len(i['seq']))

        plt.hist(length, bins=100, range=[0,maxlen])
        plt.xlabel("seq length")
        plt.ylabel("number")
        plt.savefig(name+".eps", format='eps')
        plt.show()




# interger encoding and padding
def to_int(seq, hparam):
	vec=[]
	for i in seq:
		vec.append(AA_indx[i])
	for i in range(len(vec), hparam['MAXLEN']):
		vec.append(0)	
	return np.array(vec)



def to_onehot(seq, hparam, start=0):
    onehot = np.zeros((hparam['MAXLEN'], hparam['vocab_size']), dtype=np.int32)
    l = min(MAXLEN, len(seq))
    for i in range(start, start + l):
        onehot[i, AAINDEX.get(seq[i - start], 0)] = 1
    onehot[0:start, 0] = 1
    onehot[start + l:, 0] = 1
    return onehot


def to_int1(seq, hparam):

    out =  np.zeros(hparam['MAXLEN'], dtype=np.int32)
    for i in range(len(seq)):
        out[i] = AAINDEX[seq[i]]

    return out
