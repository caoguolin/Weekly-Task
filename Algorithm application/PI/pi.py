##计算pi值
import math
import random

##首先最简单的方法自然是直接打值
##method1

def pi1():
    res = math.pi

    return res


##第二种方法用蒙特卡洛算法
##具体的做法是往一个正方形内随机丢足够多的散点，根据落在其内切圆内的散点和总散点数的比值来求pi
##是一种几何概型
##method2

def pi2(n):
    '''
    n为散点个数
    '''
    num = 0
    for i in range(n):
        x = random.random()
        y = random.random()
        if x*x + y*y <=1:
            num += 1
        
    res = (num/n)*4
    return res


##第三种方法是用级数的方法来进行求解
##pi的幂级数求和公式为:
##pi = ∑[1/16^n((4/8n+1)-(2/8n+4)-(1/8n+5)-(1/8n+6))]
##method3

def pi3(n):
    '''
    n即为上述公式里的n,为最后一项
    '''
    res = 0
    for i in range(n):
        res += (1/pow(16,i))*(4/(8*i+1)-2/(8*i+4)-1/(8*i+5)-1/(8*i+6))

    return res


##第四种方法是中国最开始求出圆周率的方法——割圆法
##这里结合隔圆术的原理图理解，图放在文件夹中
##简单来说，割圆术就是假设有一个边长为1的正六边形以及其外接圆，我们每次将正六边形的边边长两个边，这样正n边形就边长正2n边形
##当n足够大时，那么正2n边形就近似可以看做是圆了
##此时有2*pi*r=k*a
##其中r为外接圆的半径，这里一直没变都是1（开始的正六边形）,而k为里面的正多边形的边数，a为其边长
##所以pi=k*a/2*r=k*a/2
##关键在于每次割圆的过程中怎么求解出下一次的边长，这里应用的是两次勾股定理，结合图片更好理解

##method4
def pi4(n):
    '''
    n为割圆次数
    '''
    ##定义一个由当前边长求下次边长的函数方法，也是割圆术的关键
    def side(x):
        '''
        x为当前的边长
        '''
        l1 = math.sqrt(1-(x/2)**2)
        l2 = 1 - l1
        l3 = math.sqrt(l2**2 + (x/2)**2)

        return l3
    
    ##设置初值为边长为1的正六边形
    a = 1
    k = 6
    for i in range(n):
        a = side(a)
        k *= 2

    res = a*k/2
    return res
    


##在实现了以上4种方法后，接下来对它们进行评测，即迭代时间与结果精度，哪种方法能用最短的迭代时间求出最高精度的结果
##由于method1的特殊性，这里就不对其作评测，仅对后三种进行比较

import time
import matplotlib.pyplot as plt
import matplotlib

##取不同的迭代次数
nn1 = [100,1000,10000,100000]
nn2 = [10,100,500,1000]
nn3 = [10,50,100,500]
##由于三种方法的迭代原理不一样，当迭代次数过大时，可能会出现卡死或溢出现象，所以分别设置迭代区间

result1 = []
result2 = []
result3 = []
for i in range(len(nn1)):
    testn1 = nn1[i]
    testn2 = nn2[i]
    testn3 = nn3[i]
    t1 = time.clock()
    res1 = pi2(testn1)
    t2 = time.clock()
    res2 = pi3(testn2)
    t3 = time.clock()
    res3 = pi4(testn3)
    t4 = time.clock()

    tt1 = t2 - t1
    tt2 = t3 - t2
    tt3 = t4 - t3

    dic1 = dict(iter=testn1,val=res1,time=tt1)
    dic2 = dict(iter=testn2,val=res2,time=tt2)
    dic3 = dict(iter=testn3,val=res3,time=tt3)
    
    result1.append(dic1)
    result2.append(dic2)
    result3.append(dic3)


# te = open('piresult.txt','w')

# print(result1,file=te)
# print(result2,file=te)
# print(result3,file=te)

##通过画图来比较结果
xx1,yy1,xx2,yy2,xx3,yy3 = [],[],[],[],[],[]
for i in range(len(result1)):
    xx1.append(result1[i]['time'])
    yy1.append(result1[i]['val'])
    xx2.append(result2[i]['time'])
    yy2.append(result2[i]['val'])
    xx3.append(result3[i]['time'])
    yy3.append(result3[i]['val'])


# plt.plot(xx1,yy1,label='method2',color='b')
# plt.plot(xx2,yy2,label='method3',color='g')
plt.plot(xx3,yy3,label='method4',color='k')
plt.xlabel('Iteration time')
plt.ylabel('Pi')
plt.title('The relationship between accuracy and iteration time')
plt.legend()

# plt.savefig('allpi.png',dpi=200)
# plt.savefig('pi2.png',dpi=200)
# plt.savefig('pi3.png',dpi=200)
plt.savefig('pi4.png',dpi=200)
plt.show()


    






