import torch
import numpy as np
import pickle
from transformers import BertTokenizer, BertModel
import time

start = time.time()

device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('./../../bert-base-chinese')
model = BertModel.from_pretrained('./../../bert-base-chinese',output_attentions=True)
model.eval()
model.to(device)


#把 左传 读入后，并处理成各个分句
with open("ACC1.0_utf8.txt", 'r', encoding='utf-8') as f1:
    lines = f1.readlines()

results =[]
text = ''
for line in lines:
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



#通过attention head 计算字同现文件
def get_attention(text):
    sent = '[CLS]' + text + '[SEP]'
    str_tokenized_sents = tokenizer.tokenize(sent)
    if len(str_tokenized_sents)-2 != len(text):
        return 'none'
    else:
        indexed_tokens = tokenizer.convert_tokens_to_ids(str_tokenized_sents)
        tokens_tensor = torch.tensor([indexed_tokens]).to(device)
        with torch.no_grad():
            outputs = model(tokens_tensor)
            attmap = outputs[-1]

        new_attmap = np.zeros((12, 12, len(text)+2, len(text)+2))
        for layer in range(12):
            for head in range(12):
                attention = attmap[layer][0,head,:,:].cpu().numpy()
                new_attmap[layer][head] = attention
        return new_attmap


#存储各个句子的attention map
attention_map = []
sentences = []
for line in results:
    attmap = get_attention(line)
    if attmap != 'none':
        attention_map.append(attmap)
        sentences.append(line)
   
    
#输出layer-head字同现网络文件
for layer in range(12):
    for head in range(12):
        results = []
        filename ='./char_text/'+'char'+str(layer+1)+'-'+str(head+1)+'.txt'

        for k in range(len(attention_map)):
            att_map = attention_map[k][layer][head]
            att_map  = att_map [1:-1, 1:-1]
            ids = np.argmax(att_map, axis=-1)

            sentence = sentences[k]

            if len(ids) != len(sentence):
                continue

            for i in range(len(sentence)):
                char1 = sentence[i]
                char2 = sentence[ids[i]]
                results.append(char1+'\t'+char2)
            
        with open(filename, 'w', encoding='utf-8') as fp:
            for result in results:
                fp.write(result+'\n')

print(time.time()-start)

        
