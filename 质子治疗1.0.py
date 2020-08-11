import numpy as np
import matplotlib.pyplot as plt
import math
pi = 3.1415

P , a = 1.5 , 0.0019 # 拟合参数P与能量大小有关，a与吸收能力有关（取水中）
density = 1 #介质密度 默认水为1
e = 20 # MeV 该次照射选用的照射能量
layers = 20 #等能量层数
R = a*(e**2) #cm e对应的最大射程
da , db= R/2 , R  #使SOBP平滑的范围
start, end = 0, 0.8
deta = (db-da)/layers #等能量层间距
print(R)

def paint_1(x, y, z):
    '''绘图函数，传入x,y坐标，z格式
    
        参数：x横坐标，numpy数组
              y纵坐标，numpy数组
              z格式   
    '''
    plt.plot (x , y, z)
    
num = 100 # 绘图点采样
d = np.linspace(start, end , num, endpoint=True) #射程内点采样用于绘图，自变量
d_a = np.linspace(0, da , num, endpoint=True) #使SOBP平滑的范围下限
d_b = np.linspace(da, db, num, endpoint=True)   #使SOBP平滑的范围上限
D_0 = np.ones(num) #计划剂量

# 布拉格峰曲线数组形式函数
def BP(d,R):
    D_BP = np.where(np.logical_and(d>=0, d<=R) , 1/(density*P*a**(1/P)*(R-d)**(1-(1/P))) , 0)#布拉格峰曲线
    return D_BP
plt.plot (d , BP(d,R) ,"r") # 绘制布拉格峰曲线

# SOBP的加权计算函数，i为不同能量层编号，最末端编号i=0.  deta为能量层间距
def Weight(i):
    if i == 0:
        W = (density*D_0*(P**2)*(a**(1/P))*math.sin(pi/P)/(pi*(P - 1)))*((deta/2)**(1 - 1/P))
    else:
        W = (density*D_0*(P**2)*(a**(1/P))*math.sin(pi/P)/(pi*(P - 1)))*((i*deta + deta/2)**(1 - 1/P) - (i*deta - deta/2)**(1 - 1/P))
    return W

# 文献p=1.5参数下的数组形式SOBP函数
def specialSOBP():
    r = ((da - d)/(db - da))**(1/3)
    D_SOBP = D_0*(3/4+((3**(1/2))/(4*pi))*np.log(((1 + r)**2)/(1 - r + r**2)) - (3/(2*pi))*(np.arctan((2*r - 1)/(3**(1/2)))))
    D = np.where(d<=da, D_SOBP,D_0)
    return D
#paint_1 (d,specialSOBP(),"y--") 

def BP_Weight():
    for i in range(0,layers):
        RR = db - i*deta
        dd = np.linspace(start, RR , num, endpoint=True)
        bp = BP(dd,RR)*Weight(i)
        plt.plot (dd,bp,"b")

#BP_Weight()

'''def SOBP():
    s = 0
    for i in range(0,layers):
        RR = db - i*deta
        dd = np.linspace(start, RR , num, endpoint=True)
        bp = BP(dd,RR)*Weight(i)
        s = s + bp
    return s
plt.plot (d,SOBP(),"y-")'''

plt.show()
