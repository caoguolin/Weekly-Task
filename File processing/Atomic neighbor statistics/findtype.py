###对于MoS2结构体系，A_(x)B_(1-x)S2, 的alloy， 这里有不同AB分布的两百多个POSCAR
###需要在里面统计：A元素具有相同最近邻B的个数，比如 A周围有一个B的 有10个....

import numpy as np
import os
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib


######################
def getindex(a):
    b = 0
    if round(a,2) == 0.17:
        b = 1
    elif round(a,2) == 0.33:
        b = 2
    elif round(a,2) == 0.50:
        b = 3
    elif round(a,2) == 0.67:
        b = 4
    elif round(a,2) == 0.83:
        b = 5
    elif round(a,2) == 1.00:
        b = 6
    
    return(b)
        
########################
def near(a):
    index1 = (a[0],a[1]+2)
    index2 = (a[0]-1,a[1]+1)
    index3 = (a[0]+1,a[1]+1)
    index4 = (a[0]-1,a[1]-1)
    index5 = (a[0]+1,a[1]-1)
    index6 = (a[0],a[1]-2)
    index = (index1,index2,index3,index4,index5,index6)
    index = np.array(index)
    index[index==0] = 6
    index[index==-1] = 5
    index[index==7] = 1
    index[index==8] = 2

    return(index)
    
########################
def getout(B,index):
    zrind = []
    for i in range(len(B)):
        posy = float(B[i].split()[1])
        posz = float(B[i].split()[2])
        indexy = getindex(posy)
        indexz = getindex(posz)
        zrind.append((indexy,indexz))
        index[indexy-1][indexz-1] = 1

    kinds = []
    for i in range(len(zrind)):
        gg = near(zrind[i])
        kind = []
        for j in range(len(gg)):
            key = gg[j]
            kind.append(index[key[0]-1][key[1]-1])
    
        num_count = Counter(kind)
        kinds.append(num_count)

    return(kinds)

##########################
def sor(a):
    b0,b1,b2,b3,b4,b5,b6=0,0,0,0,0,0,0
    for i in range(len(a)):
        m = a[i]
        if m[0] == 0:
            b0+=1
        elif m[0] == 1:
            b1+=1
        elif m[0] == 2:
            b2+=1
        elif m[0] == 3:
            b3+=1
        elif m[0] == 4:
            b4+=1
        elif m[0] == 5:
            b5+=1
        elif m[0] == 6:
            b6+=1
    
    return(b0,b1,b2,b3,b4,b5,b6)

###########################
def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            yield fullname

###########################

filename = [3,4,5,6,7,8,9,10,11,12,13,14,15]
message = open('message.txt','w')
mes = open('mes.txt','w')

print("这是每个结构的具体信息:",file=message)
print("在每个族群中，每一个数组的含义表示此结构中Zr原子周围n个Hf原子时的个数，n取0，1，2，3，4，5，6，对应每个数组的7个元素",file=mes)
print("由于结构中每个原子的最近邻有6个原子（一个圆形），这里知道某Zr原子周围Hf原子数目，也就知道了其周围Zr原子数目",file=mes)
print("每个族群的数组个数则代表此族群共有多少个POSCAR文件，如(0, 0, 0, 0, 0, 2, 1)则表示此结构中周围有5个Hf和1个Zr的Zr有2个，周围有6个Hf的Zr有1个",file=mes)
message.write("\r\n")
mes.write("\r\n")

m0,m1,m2,m3,m4,m5,m6=0,0,0,0,0,0,0

for i in range(len(filename)):
    path = []
    base = './find/str2/{}Zr'.format(filename[i])
    print("族群{},Zr原子个数为{}:".format(i+1,filename[i]),file=message)
    print("族群{},Zr原子个数为{}:".format(i+1,filename[i]),file=mes)
    for j in findAllFile(base):
        path.append(j)

    for k in range(len(path)):
        a = open(path[k],'r')
        aa = a.readlines()
        ele = aa[5].split()
        num = aa[6].split()
        if ele[0] == 'Hf':
            A = aa[8:8+int(num[0])]
            B = aa[8+int(num[0]):8+int(num[0])+int(num[1])]
        else:
            B = aa[8:8+int(num[0])]
            A = aa[8+int(num[0]):8+int(num[0])+int(num[1])]
        
        index = np.zeros((6,6))
        mmm = getout(B,index)
        nnn = sor(mmm)
        m0+=nnn[0]
        m1+=nnn[1]
        m2+=nnn[2]
        m3+=nnn[3]
        m4+=nnn[4]
        m5+=nnn[5]
        m6+=nnn[6]
        print("当前结构Zr原子个数："+str(len(B)),file=message)
        print(nnn,file=mes)
        for l in range(7):
            print("当前结构中有Zr周围{}个Hf的Zr个数为：".format(l)+str(nnn[l]),file=message)
    
    message.write("\r\n")
    mes.write("\r\n")

message.close()
mes.close()


################画图
y = [m0,m1,m2,m3,m4,m5,m6]
x = range(0,7)

plt.rcParams['font.sans-serif']=['simhei']
plt.bar(x,y)
plt.xlabel('Zr原子周围Hf原子数目')
plt.ylabel('此类Zr原子数目')
plt.title('Zr原子种类统计')

for x,y in zip(x,y):
    plt.text(x,y+0.1,'%s' %y,ha='center')

plt.plot()
plt.savefig('result.png', dpi=200)       
plt.show()   
