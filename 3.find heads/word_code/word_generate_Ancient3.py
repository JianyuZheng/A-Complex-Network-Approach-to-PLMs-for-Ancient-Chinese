#让一句话里的词统一关注末词
import torch
import numpy as np
import pickle
from transformers import BertTokenizer, BertModel
import time

start = time.time()

device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('./../../bert-siku-chinese')
model = BertModel.from_pretrained('./../../bert-siku-chinese',output_attentions=True)
model.eval()
model.to(device)

#把HIT-all读入后，并处理成各个分句
with open("ACC1.0_utf8.txt", 'r', encoding='utf-8') as f1:
    lines = f1.readlines()



results =[]
text = ''
for line in lines:                    #改一下
    line = line.strip().split()
    for token in line:
        word = token.split('/')[0]
        tag = token.split('/')[1]
        if tag == 'w':
            results.append(text.strip())
            text = ''
        else:
            text = text+' '+word
    if text !='':
        results.append(text.strip())
        text = ''
results = [result for result in results if result.strip() != '']


newlines = []

seg_ids = []
for result in results:
    newline = ''
    seg_id = [0,1]
    id = 1
    result = result.strip().split()

    for word in result:
        newline += word
        id += len(tokenizer.tokenize(word))
        seg_id.append(id)
    seg_id.append(id+1)
    seg_ids.append(seg_id)
    newlines.append(newline)



new_results = []
for i in range(len(results)):
    sent = newlines[i]
    sent = '[CLS]' + sent + '[SEP]'
    str_tokenized_sents = tokenizer.tokenize(sent)
    if len(str_tokenized_sents) != seg_ids[i][-1]:
        continue
    else:
        newline = results[i].strip().split()
        for j in range(len(newline)):
              word1 = newline[j]
              word2 = newline[-1]
              new_results.append(word1+'\t'+word2)

print(len(new_results)-3)
with open('word_generate_Ancient3.txt', 'w', encoding='utf-8') as fp:
    for line in new_results[3:]:
        fp.write(line+'\n')
        
        
        
