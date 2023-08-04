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

#把HIT-all读入后，并处理成各个分句
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



#通过attention head 计算词同现文件
def get_attention(text, seg_id):
    sent = '[CLS]' + text + '[SEP]'
    str_tokenized_sents = tokenizer.tokenize(sent)
    if len(str_tokenized_sents) != seg_id[-1]:
        return 'none'
    else:
        indexed_tokens = tokenizer.convert_tokens_to_ids(str_tokenized_sents)
        tokens_tensor = torch.tensor([indexed_tokens]).to(device)
        with torch.no_grad():
            outputs = model(tokens_tensor)
            attmap = outputs[-1]

        new_attmap = np.zeros((12, 12, len(seg_id)-1, len(seg_id)-1))
        for layer in range(12):
            for head in range(12):
                attention = attmap[layer][0,head,:,:].cpu().numpy()
                re_attention = np.zeros((len(seg_id)-1, len(seg_id)-1))
                for i in range(len(seg_id)-1):
                    for j in range(len(seg_id)-1):
                        re_attention[i][j] = np.sum(attention[seg_id[i]:seg_id[i+1], seg_id[j]:seg_id[j+1]])/(seg_id[i+1]-seg_id[i])
                new_attmap[layer][head] = re_attention
        return new_attmap


#存储各个句子的attention map
attention_map = []
sentences = []
for i in range(len(newlines)):
    line = newlines[i]
    seg_id = seg_ids[i]

    attmap = get_attention(line, seg_id)
    if attmap != 'none':
        attention_map.append(attmap)
        sentences.append(line)
   
    
#输出layer-head词同现网络文件
for layer in range(12):
    for head in range(12):
        results = []
        filename ='./word_text/'+'word'+str(layer+1)+'-'+str(head+1)+'.txt'

        for k in range(len(attention_map)):
            att_map = attention_map[k][layer][head]
            att_map  = att_map [1:-1, 1:-1]
            ids = np.argmax(att_map, axis=-1)

            sentence = sentences[k]
            seg_id = seg_ids[k][1:-1]
            seg_id = [id-1 for id in seg_id]

            for i in range(len(ids)):
                word1 = sentence[seg_id[i]:
                                 seg_id[i+1]]
                word2 = sentence[seg_id[ids[i]]:
                                 seg_id[ids[i]+1]]
                results.append(word1+'\t'+word2)
            
        with open(filename, 'w', encoding='utf-8') as fp:
            for result in results[3:]:
                fp.write(result+'\n')

            
            
        
