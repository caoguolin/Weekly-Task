import numpy as np
import random
import os
import math

'''
本例中的体系为PtNi合金，合金6层slab
最上面一层和最下面一层为纯Pt，中间四层为Pt-Ni合金（类似Pt皮肤）
原生Pt结构为56个Pt原子，替换其中部分的Pt原子为Ni进而生成目标结构
'''

xx = 0.3
numpos = 300
numatom = 36
'''
以上为调用参数
xx为替换的金属的比例，即AxB1-x中的A的比例
numpos为生成的POSCAR总数
numatom为替换部分的原子总数
'''

def rep(x,a,b):
    '''
    x为替换的金属的比例，即假如AxB1-x中A为替换的金属
    a为要替换部分的原子总数
    b为随机生成的POSCAR文件数
    '''
    m = range(1,a+1)
    n = []
    for i in range(1,1001):
        mm = random.sample(m,int(a*x))
        mm.sort()
        n.append(mm)

    res = random.sample(n,b)

    return res

with open("Pt.vasp",'r') as fi1:
    flist1 = fi1.readlines()
    num = rep(xx,numatom,numpos)
    atoms = flist1[17:53]
    pt1 = flist1[53:]
    for i in range(len(num)):
        num0 = num[i]
        atompt = []
        atomni = []
        for j in range(len(atoms)):
            if j+1 in num0:
                atomni.append(atoms[j])
            else:
                atompt.append(atoms[j])

        atomni
        atompt
        aaa = 26+numatom-int(numatom*xx)
        for l in range(len(pt1)):
            flist1[l+17] = pt1[l]

        for k in range(numatom-int(numatom*xx)):
            flist1[k+26] = atompt[k]

        for m in range(int(numatom*xx)):
            flist1[26+numatom-int(numatom*xx)+m] = atomni[m]

        flist1[5] = "Pt Ni \n"
        flist1[6] = "{} {} \n".format(54-int(numatom*xx),int(numatom*xx))

        os.makedirs("res111/{}/".format(i+1))
        with open("res111/{}/POSCAR".format(i+1),"w") as fi:
            fi.writelines(flist1)
