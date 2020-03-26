amino_acid = ['A', 'R', 'N', 'D',  'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
AAINDEX = {}
import numpy as np

for i in range(len(amino_acid)):
    AAINDEX[amino_acid[i]] = i + 1
    
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
def to_intger(seq):
    vec = np.zeros((MAXLEN), dtype=np.int32)
    for i in range(len(seq)):
        vec[i] = np.where(seq[i], amino_acid)+1
    
    return vec


def to_onehot(seq, hparam, start=0):
    onehot = np.zeros((hparam['MAXLEN'], hparam['vocab_size']), dtype=np.int32)
    l = min(MAXLEN, len(seq))
    for i in range(start, start + l):
        onehot[i, AAINDEX.get(seq[i - start], 0)] = 1
    onehot[0:start, 0] = 1
    onehot[start + l:, 0] = 1
    return onehot


def to_int(seq, hparam):

    out =  np.zeros(hparam['MAXLEN'], dtype=np.int32)
    for i in range(len(seq)):
        out[i] = AAINDEX[seq[i]]

    return out