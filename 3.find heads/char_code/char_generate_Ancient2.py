#让一句话里的字统一关注句首字
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
for line in lines:                   #改一下
    line = line.strip().split()
    for token in line:
        word = token.split('/')[0]
        tag = token.split('/')[1]
        if tag == 'w':  
            results.append(text)
            text = ''
        else:
            text += word
    if text !='':
        results.append(text)
        text = ''
results = [result for result in results if result.strip() != '']

new_results = []

for i in range(len(results)):
    sent = results[i]
    sent = '[CLS]' + sent + '[SEP]'
    str_tokenized_sents = tokenizer.tokenize(sent)
    if len(str_tokenized_sents)-2 != len(results[i]):
        continue
    else:
        for j in range(len(results[i])):
              char1 = results[i][j]
              char2 = results[i][0]
              new_results.append(char1+'\t'+char2)

print(len(new_results))        
with open('char_generate_Ancient2.txt', 'w', encoding='utf-8') as fp:
    for line in new_results:
        fp.write(line+'\n')

        
        
