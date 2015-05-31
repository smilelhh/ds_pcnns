import numpy as np
import cPickle
import time, os

class InstanceBag(object):

    def __init__(self, entities, rel, num, sentences, positions, entitiesPos):
        self.entities = entities
        self.rel = rel
        self.num = num
        self.sentences = sentences
        self.positions = positions
        self.entitiesPos = entitiesPos

def readData(filename):
    f = open(filename, 'r')
    data = []
    while 1:
        line = f.readline()
        if not line:
            break
        entities = map(int, line.split(' '))
        line = f.readline()
        bagLabel = line.split(' ')

        rel = map(int, bagLabel[0:-1])
        num = int(bagLabel[-1])
        positions = []
        sentences = []
        entitiesPos = []
        for i in range(0, num):
            sent = f.readline().split(' ')
            positions.append(map(int, sent[0:2]))
            epos = map(int, sent[0:2])
            epos.sort()
            entitiesPos.append(epos)
            sentences.append(map(int, sent[2:-1]))
        ins = InstanceBag(entities, rel, num, sentences, positions, entitiesPos)
        data += [ins]
    f.close()
    return data

def wv2pickle(filename='wv.txt', dim=50, outfile='Wv.p'):
    f = open(filename, 'r')
    allLines = f.readlines()
    f.close()
    #tmp = allLines[0]
    Wv = np.zeros((len(allLines)+2, dim))
    i = 1
    for line in allLines:
        #tmp = map(float, line.split(' '))
        #Wv[i, :] = tmp
        Wv[i, :] = map(float, line.split(' '))
        i += 1

    rng = np.random.RandomState(3435)
    #tmp = rng.uniform(low=-0.5, high=0.5, size=(1, dim))
    Wv[i, :] = rng.uniform(low=-0.5, high=0.5, size=(1, dim)) #my unknown embedding
     #save Wv
    f = open(outfile, 'w')
    cPickle.dump(Wv, f, -1)
    f.close()

def data2pickle(input, output):
    data = readData(input)
    f = open(output, 'w')
    cPickle.dump(data, f, -1)
    f.close()
class ProgressBar():
    def __init__(self, width=50):
        self.pointer = 0
        self.width = width

    def __call__(self,x):
         # x in percent
         self.pointer = int(self.width*(x/100.0))
         return "|" + "#"*self.pointer + "-"*(self.width-self.pointer)+\
                "|\n %s percent done" % str(x)


if __name__ == "__main__":
    # data = readData('train_filtered_len_60_gap_40.data')
    #wv2pickle('wv.txt', 50, 'Wv1.p')
    data2pickle('test_filtered_len_60_gap_40.data', 'test_len_60_gap_40.p')
    data2pickle('train_filtered_len_60_gap_40.data', 'train_len_60_gap_40.p')
    # pb = ProgressBar()
    # print pb(0.5)
    # for i in range(101):
    #     os.system('clear')
    #     print pb(i)
    #     time.sleep(0.1)
    # for i in range(101):
    #     os.system('clear')
    #     print pb(i)
    #     time.sleep(0.1)

