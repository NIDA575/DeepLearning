import numpy as np
import math


step = 4/128;
print(step)
y=[]
y_fi=[]
for i in range(0,128):
    temp=math.tanh(i*step)
    y.append(temp)
#print(y)

for i in range(len(y)):
    x=(np.round(y[i]*127))
    y_fi.append(x)

print(y_fi)

p= [1,34,-45,-67,127,-98]

for i in range(len(p)):
    if p[i]<0:
        res=y_fi[-1*p[i]]
        res=-1*res
        print(f'{p[i]}={res}')
    else:
        res=y_fi[p[i]]
        print(f'{p[i]}={res}')

exit(0)
import matplotlib.pyplot as plt
plt.plot(y)
plt.show()
exit(0)

y=math.tanh(step)

print(y)

