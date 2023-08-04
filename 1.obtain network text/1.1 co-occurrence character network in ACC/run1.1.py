


with open('Dataset_ACC1.0.txt', 'r', encoding='utf-8') as f1:
    lines = f1.readlines()
lines = [line.strip() for line in lines]


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

texts = []
for result in results:
    for i in range(len(result)-1):
        texts.append(result[i]+ '\t' + result[i+1])

with open('char_occurence.txt','w', encoding='utf-8') as fp:
    for text in texts:
        fp.write(text+'\n')
