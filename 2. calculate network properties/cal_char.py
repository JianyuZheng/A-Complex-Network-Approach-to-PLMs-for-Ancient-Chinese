#该脚本用于计算所处理文本经复杂网络表示后的相关参数
#记得改输入文件的名字 和存储文件的名字
import networkx as nx
import numpy as np 
import matplotlib.pyplot as plt
import time
import pickle



def run(filename):
    G1=nx.Graph()  #连通图

    #打开文件
    with open(filename, 'r', encoding='utf-8') as f1:
        lines = f1.readlines()

    #构造图
    for line in lines:
        G1.add_edge(line.strip().split()[0], line.strip().split()[1])

    N = G1.number_of_nodes()  #节点数
    E = G1.number_of_edges()  #边数
    print(N,E)
    degs = list(G1.degree())
    x = 0
    for deg in degs:
        x += deg[1]
    k_ave = x/len(degs)  #平均度

    def D_L(G):
        dd = []
        pair_d = nx.shortest_path_length(G)
        for p in pair_d:
            dd += list(p[1].values())[1:]
        return (np.max(dd), np.mean(dd))  #返回(直径, 平均最短路径长度)
    D, L = D_L(G1)       #平均最短路径长度

    cc = nx.clustering(G1)
    C = sum(cc.values())/len(cc)  #平均聚类系数

    density = nx.density(G1)   #稠密度

    #求各种中心性
    #https://networkx.github.io/documentation/networkx-1.9/reference/generated/networkx.algorithms.centrality.betweenness_centrality.html
    #度数中心度(Degree Centrality)不用算，跟平均度差不多
    #接近中心度(Closeness Centrality)不用算，跟平均最短路径长度差不多
    Bc = nx.betweenness_centrality(G1)        #中介中心度(Betweenness Centrality)
    Bc_ave = sum(Bc.values())/N
    Ec = nx.eigenvector_centrality(G1)        #特征向量中心度(Eigenvector Centrality)
    Ec_ave = sum(Ec.values())/N
    #with open('Bc.dat','wb') as f:
        #pickle.dump(Bc,f)


    #拟合度分布
    degree = [deg[1] for deg in degs]
    distKeys = set(degree)
    pdf = dict([(k,0) for k in distKeys])
    for k in degree:
        pdf[k] += 1

    cdf = dict([(k,0) for k in distKeys])
    for k in set(degree):
        cdf[k] = sum(np.array(degree)>=k)

    #求累计的幂律分布
    #要去log双对数的原因 https://blog.csdn.net/Eastmount/article/details/65443025
    x = list(cdf.keys())

    if 0 in x:
        x.remove(0)
        S = sum(cdf.values())-cdf[0]
    else:
        S = sum(cdf.values())
    y = np.array([cdf[k]/S for k in x])
    x_log = np.log10(np.array(x))
    y_log = np.log10(y)
    A = np.vstack([x_log, np.ones(len(x_log))]).T
    gama,c = np.linalg.lstsq(A,y_log)[0]
    RSS= np.linalg.lstsq(A,y_log)[1] #残差总和
    TSS = np.sum(np.square(y_log-np.mean(y_log))) #http://blog.sina.com.cn/s/blog_17bf54ea20102x70y.html
    R2 = list((TSS-RSS)/TSS)       #拟合系数
    if R2 != []:
        R2 = R2[0]
    else:
        R2 = 0


    #plt.plot(x_log, y_log, 'ro', linewidth=0.0005)
    #plt.plot(x_log, gama*x_log+c, 'k-',label=r'$\gamma=%s$'%(round(gama,2)))
    #plt.savefig(filename[:-4]+'png', dpi=1000)
    #plt.show()
    return [N, E, k_ave, gama, R2, L, C, D, density, Bc_ave, Ec_ave]

    '''
    print("N:", N)                 #节点数
    print("E:", E)                 #边数
    print("<k>:", k_ave)           #平均度
    print("r:", gama)              #幂律分布的指数
    print("R2:", R2)               #幂律分布拟合系数
    print("L:", L)                 #平均最短路径长度
    print("C:", C)                 #平均聚集系数
    print("D:", D)                 #网络直径
    print("density:", density)     #稠密度
    print("Bc_ave", Bc_ave)        #中介中心性
    print("Ec_ave", Ec_ave)        #特征向量中心性
    '''



filename = './char_occurence.txt'
result = run(filename)


#写入pickle
with open('char_occurence.dat', 'wb') as f:
    pickle.dump(result, f)




