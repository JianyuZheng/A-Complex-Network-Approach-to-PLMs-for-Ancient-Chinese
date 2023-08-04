#当前字关注词末字
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
            results.append(text.strip())
            text = ''
        else:
            text = text+' '+word
    if text.strip() !='':
        results.append(text.strip())
        text = ''
results = [result for result in results if result.strip() != '']


new_results = []
for i in range(len(results)):
    sent = results[i].replace(' ','')
    sent1 = results[i].replace(' ','')
    sent = '[CLS]' + sent + '[SEP]'
    str_tokenized_sents = tokenizer.tokenize(sent)
    if len(str_tokenized_sents)-2 != len(sent1):
        continue
    else:
        newline = results[i].strip().split()
        for j in range(len(newline)):
            for k in range(len(newline[j])):
                char1 = newline[j][k]
                char2 = newline[j][-1]
                new_results.append(char1+'\t'+char2)


print(len(new_results))
with open('char_generate_Ancient4.txt', 'w', encoding='utf-8') as fp:
    for line in new_results:
        fp.write(line+'\n')
        
        
        
