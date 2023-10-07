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
print("Example Input : king-queen-prince")

# Main loop for analogy

while True:
   
    analogy_input = input("\nEnter three words and put " +"'-'"+ " in between every word (EXIT to break): ").lower()
    
    if analogy_input.upper() == 'EXIT':
        break
    else:
        new_input= analogy_input.lower().split("-")
        np_first_word=np.array(vectors[new_input[0]])
        np_second_word=np.array(vectors[new_input[1]])
        new_subtracted_array_for_first_and_word=np_second_word-np_first_word
        thrid_word=new_input[2]
        np_thrid_word=np.array(vectors[thrid_word])
        forth_word=np_thrid_word+new_subtracted_array_for_first_and_word
        subtractable_forth_word=np.tile(forth_word,(400000,1))
        subtractted_froth_word=W-subtractable_forth_word
        squer_subtractted_froth_word=np.square(subtractted_froth_word)
        forth_word_distance=[]
        for i in range(0,len(squer_subtractted_froth_word)):
            forth_word_distance.append(sum(squer_subtractted_froth_word[i]))
        np_arraya_distince_for_froth_word=np.array(forth_word_distance)
        np_arraya_distince_for_froth_word=np.sqrt(np_arraya_distince_for_froth_word)
        forth_word_sorted_indices = np.argsort(np_arraya_distince_for_froth_word)
        forth_word_smallest_values = np_arraya_distince_for_froth_word[forth_word_sorted_indices[:5]]
        forth_word_smallest_indices = forth_word_sorted_indices[:5]
        print("The expected words are:")
        count=0 
        for i in forth_word_smallest_indices:
            if ivocab[i] != new_input[0] and ivocab[i] != new_input[1] and ivocab[i] != new_input[2]:
                print(ivocab[i])
                count=count+1
        
            if count==2:
                break