import torch
import numpy as np
x=np.array([0,9,5,4,7,8,9])
"""
y=['4','6','6','6','6','6','6']
z=['3','5','8']
xx=[x,y,z]
def fun(x):
    return abs(int(x)-10)
# sorted函数
kk=sorted(x,key=fun,reverse=True)
print(kk)
print(x)
#emunate函数
for i,num in enumerate(y,2):
    print('序号{}内容{}'.format(i,num))
##all于any用法
x=[ temp-4 for temp in x  ]
if any(x):
    print(x)
else:
    print("hhh{}".format(x))

x=np.array([0,9,5,4,7,8])

x=torch.from_numpy(x)
y=x-4
z=['sxd','sxd','sxd','sxd','sxd','sxd']
print(x)
print(y)
print(z)

xx=zip(x,y,z)
print(xx)
for i,(x1,x2,x3) in enumerate(xx):
    print("序号{}   X{}   Y{}   Z{}".format(i,x1,x2,x3))

"""

from tqdm import tqdm
import time
phar=tqdm(range(1000))
set=torch.randn(1000)
def log_msg(msg,mode):
    dict={
        "train": 34,
        "eval": 33
    }

    msg=" \033[{}m[{}] {}\033[0m".format(dict[mode],mode,msg)
    return msg
for i in set:
    time.sleep(0.01)
    msg=i
    phar.set_description(log_msg(msg,"train"),True)
    phar.update()
phar.close()








