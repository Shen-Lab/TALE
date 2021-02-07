import pickle
import os
import argparse
from Utils import metric
import numpy as np

def gen_train_dnmd(seq, train_fasta):

   with open(train_fasta, "w") as f:
      for i in range(len(seq)):
              f.write('>'+str(i)+'\n')
              f.write(seq[i]['seq']+'\n')
   
   os.system("../diamond/diamond makedb --in "+train_fasta+" -d "+train_fasta)


def predict(fasta, train_dnmd, test_seq, train_seq,  nlabels):
  
   output = os.popen("../diamond/diamond blastp -d "+train_dnmd+" -q "+fasta+" --outfmt 6 qseqid sseqid bitscore").readlines() 

   test_bits={}
   test_train={}
   for lines in output:
       line = lines.strip('\n').split()
       if line[0] in test_bits:
          test_bits[line[0]].append(float(line[2]))
          test_train[line[0]].append(line[1])
       else:
          test_bits[line[0]] = [float(line[2])]
          test_train[line[0]] = [line[1]]
       #print (lines) 
   
   preds_score=[]
   for s in range(len(test_seq)):
           probs = np.zeros(nlabels, dtype=np.float32)
           if str(s) in test_bits:
                weights = np.array(test_bits[str(s)])/np.sum(test_bits[str(s)])
                
                for j in range(len(test_train[str(s)])):
                  temp = np.zeros(nlabels)
                  #print (s, j, test_train[str(s)])
                  temp[ train_seq[int(test_train[str(s)][j])]['label']  ] = 1.0
                  probs+= weights[j]* temp
           
           preds_score.append(probs)

   return np.array(preds_score)


def main():
        

        parser = argparse.ArgumentParser(description='Arguments for pretrain_seq.py')
        parser.add_argument('--fasta', default=None, type=str)
        parser.add_argument('--on', default=None, type=str)
        parser.add_argument('--data_dir', default='../data/Gene_Ontology/CAFA3/', type=str)
        args = parser.parse_args() 

        if args.fasta == None:
            raise ValueError("Must specify a fasta file.")
        if args.on == None:
            raise ValueError("Must specify the ontology.")


        with open(args.data_dir+"/test_seq_"+args.on, "rb") as f:
            seq_test=pickle.load(f)

        with open(args.data_dir+"/train_seq_"+args.on, "rb") as f:
            seq_train=pickle.load(f)


        train_fasta = args.data_dir+'trainseq_'+args.on+".fasta"

        if not os.path.exists(train_fasta):
            gen_train_dnmd(seq_train, train_fasta)
        

        test_fasta = args.data_dir+'testseq_'+args.on+".fasta" 
        with open(test_fasta, "w") as f:
         for i in range(len( seq_test)):
               f.write('>'+str(i)+'\n')
               f.write(seq_test[i]['seq']+'\n')
          

        with open(args.data_dir+args.on+"_go_1.pickle", "rb") as f:
            graph = pickle.load(f)
            nlabels = len(graph)
            del graph

        preds = predict(test_fasta,  train_fasta+'.dmnd', seq_test, seq_train, nlabels)        

        true_label=[]

        for i in seq_test:
               true_label1 = np.zeros(nlabels, dtype=np.int32)
               true_label1[i['label']] = 1
               true_label.append(true_label1)
        true_label = np.array(true_label)

        true_labelv1 = np.load("../ResultCAFA3/truelabel_"+args.on+".npy")

        print (true_label.shape, true_labelv1.shape)
        assert np.array_equal(true_label, true_labelv1)


        print (preds.shape, true_label.shape)
        fmax, smin, auprc = metric.main(true_label, preds)

        print ("Diamond ontology:%s  fmax: %.3f  auprc: %.3f\n" %(args.on, fmax,  auprc))
        np.save("../ResultCAFA3/Diamond_"+args.on, preds)        
        #np.save("Results/truelabel_"+args.on, true_label)

if __name__ == '__main__':
        main()
