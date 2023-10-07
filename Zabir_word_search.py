import random
import numpy as np

vocabulary_file='word_embeddings.txt'

print('Read words...')
with open(vocabulary_file,'r',encoding="utf8") as f:
    words = [x.rstrip().split(' ')[0] for x in f.readlines()]

print('Read word vectors...')
with open(vocabulary_file, 'r',encoding="utf8") as f:
    vectors = {}
    for line in f:
        vals = line.rstrip().split(' ')
        vectors[vals[0]] = [float(x) for x in vals[1:]]

vocab_size = len(words)
vocab = {w: idx for idx, w in enumerate(words)}
ivocab = {idx: w for idx, w in enumerate(words)}

print('Vocabulary size:')
print(len(vocab))
print(vocab['man'])
print(len(ivocab))
print(ivocab[10])


# W contains vectors for

print('Vocabulary word vectors :')
vector_dim = len(vectors[ivocab[0]])
print(vector_dim)

W = np.zeros((vocab_size, vector_dim))

for word, v in vectors.items():
    if word == '<unk>':
        continue
    W[vocab[word], :] = v
print(W.shape)

# Main loop for analogy

while True:
    input_term = input("\nEnter the word (EXIT to break): ").lower()
    if input_term.upper() == 'EXIT':
        break
    else:
        subtractable_input_term=np.tile(vectors[input_term],(400000,1))
        subtracted_array=W-subtractable_input_term
        squared_subtracted_array = np.square(subtracted_array)
        distance=[]
        for i in range(0,len(squared_subtracted_array)):
            distance.append(sum(squared_subtracted_array[i]))

        np_array_distance=np.array(distance)
        np_array_distance=np.sqrt(np_array_distance)
        sorted_indices = np.argsort(np_array_distance)
        smallest_values = np_array_distance[sorted_indices[:3]]
        smallest_indices = sorted_indices[:3]
        print("\n                               Word         \tDistance\n")

        for i in smallest_indices:
            print("%35s\t\t%f\n" % (ivocab[i],np_array_distance[i]))