import numpy as np
import os
import math


def transform(m,a,b,c):
    '''
    该函数用于将分数坐标转化为笛卡尔坐标
    m为lattice的系数,(a,b,c)对应当前点的坐标(x,y,z)
    '''
    x0 = m[0][0]*a + m[1][0]*b + m[2][0]*c
    y0 = m[0][1]*a + m[1][1]*b + m[2][1]*c
    z0 = m[0][2]*a + m[1][2]*b + m[2][2]*c

    return(x0,y0,z0)


def la(lattice,n):
    '''
    该函数用于扩胞时修改lattice的系数
    lattice为POSCAR中lattice的三行文本
    n为扩胞时的放大系数
    '''
    la1 = lattice.split()
    la2 = []
    for i in range(2):
        la2.append(str(float(la1[i])*n))
        
    la2.append(la1[2])
    la3 = " ".join(la2) + " \n"

    return(la3)

def findAllFile(base):
    '''
    该函数用于打开分层级文件夹
    base为根目录
    '''
    for root, ds, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            yield fullname


def Supercell(position,lat,superlist=[[-1,0,1],[-1,0,1],[-1,0,1]]):
    '''
    position为所有的原子的坐标，array形式
    lat为晶格数组
    superlist为扩胞方向，对应x,y,z，默认情况下，组合共有3*3*3种情况
    '''
    ###计算超胞尺寸
    super_size = []
    for i in superlist[0]:
        for j in superlist[1]:
            for k in superlist[2]:
                super_size.append([i,j,k])
    super_size = np.array(super_size)
    super_lattice = np.dot(super_size,lat)
    

    ###计算扩胞后原子坐标
    super_position = []
    for i in range(len(position)):
        new_position = position[i] + super_lattice
        super_position.extend(new_position.tolist())
    super_position = np.array(super_position)

    return(super_position)



filename = [3,4,5,6,7,8,9,12]
for i in range(len(filename)):
    path = []
    base = './find/str2/{}Zr'.format(filename[i])
    for j in findAllFile(base):
        path.append(j)

    for k in range(len(path)):
        with open(path[k],'r') as fi1:
            flist1 = fi1.readlines()
            lat = flist1[2:5]
            poslat = np.array([i.split() for i in flist1[2:5]],dtype=float)
            flist1[7] = "Cartesian \n"
            pos = flist1[8:]
            posnew = []
            for l in range(len(pos)):
                a = float((pos[l].split())[0])
                b = float((pos[l].split())[1])
                c = float((pos[l].split())[2])
                newpos = transform(poslat,a,b,c)
                posnew.append(newpos)
                
            ####转化笛卡尔坐标完毕
            posnew = np.array(posnew)
            num = flist1[6].split()
            num0 = []
            for n in range(len(num)):
                num0.append(str(int(num[n])*9))
            flist1[6] = " ".join(num0) + " \n"
            flist1[8:] = " "

            sup = Supercell(posnew,poslat,superlist=[[-1,0,1],[-1,0,1],[0]])
            sup = sup.tolist()
            for m in range(len(sup)):
                pp = " ".join('%s'%id for id in sup[m]) + " \n"
                flist1.insert(8+m,pp)

            for w in range(2):
                flist1[w+2] = la(lat[w],3)

            ###扩胞完毕

            lal = flist1[2].split()
            lal1 = [lal[0],lal[0],str(0.0)]
            flist1[2] = " ".join(lal1) + " \n"

            ###长方形cell转化为平行四边形cell完毕

            with open(path[k].replace("POSCAR","POSCAR111"),"w") as fi:
                fi.writelines(flist1)


            
