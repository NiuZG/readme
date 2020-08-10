import numpy as np
import matplotlib.pyplot as plt
pi = 3.1415

P , a = 1.5 , 0.0019 # 拟合参数P与能量大小有关，a与吸收能力有关（取水中）
density = 1 #介质密度 默认水为1
e = 20 # MeV 该次照射选用的照射能量
layers = 20 #等能量层数
R = a*(e**2) #cm e对应的最大射程
da , db= R/2 , R  #使SOBP平滑的范围
deta = (db-da)/layers #等能量层间距
print(R)

w = [120, 100, 50, 35, 28, 25, 20, 19, 17, 16, 15, 13, 12, 11, 10, 9, 8, 9, 8, 9] #各能量层权重


num = 10000# 绘图点采样
d = np.linspace(0, R+R/num , num, endpoint=True) #射程内点采样用于绘图，自变量
d_a = np.linspace(0, da , num, endpoint=True) #使SOBP平滑的范围下限
d_b = np.linspace(da, db, num, endpoint=True)   #使SOBP平滑的范围上限

D_0 = np.ones(num) #计划剂量
D_BP = np.where(d<=R , 1/(density*P*a**(1/P)*(R-d)**(1-(1/P))) , 0)#布拉格峰曲线
print(D_BP)
plt.plot (d , D_BP ,"ro-") # 绘制布拉格峰曲线

r = ((da - d)/(db - da))**(1/3)
D_SOBP = D_0*(3/4+((3**(1/2))/(4*pi))*np.log(((1 + r)**2)/(1 - r + r**2)) - (3/(2*pi))*(np.arctan((2*r - 1)/(3**(1/2)))))
D = np.where(d<=da, D_SOBP,D_0)
plt.plot (d,D,"b") # 绘制SOBP曲线

r = ((da - d)/(db - da))**(1-1/P)
D_SOBP = D_0*(3/4+((3**(1/2))/(4*pi))*np.log(((1 + r)**2)/(1 - r + r**2)) - (3/(2*pi))*(np.arctan((2*r - 1)/(3**(1/2)))))
D = np.where(d<=da, D_SOBP,D_0)
plt.plot (d,D,"g") # 绘通用SOBP曲线

bp = 1/(density*P*a**(1/P)*(R-d)**(1-(1/P)))
s = np.zeros(num)
RR = 0
for i in range(0,layers):
    RR = db - i*deta
    dd = np.linspace(0, RR , num , endpoint=True)
    if i==0:
        W = (density*D_0*(P**2)*(a**(1/P))*np.sin(pi/P)/(pi*(P - 1)))*((deta/2)**(1 - 1/P))
    else:
        W = (density*D_0*(P**2)*(a**(1/P))*np.sin(pi/P)/(pi*(P - 1)))*((db - RR + deta/2)**(1 - 1/P) - (db - RR - deta/2)**(1 - 1/P))
    bp = np.where(d<=RR , W/(density*P*a**(1/P)*(RR - dd)**(1-(1/P))) , 0)
    plt.plot (d,bp,"b--")
    s = s + bp
   

plt.plot (d,s,"y-")

plt.show()


#weight = np.where(da<= db-deta <=R , W , 0)
