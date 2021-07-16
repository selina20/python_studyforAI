#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt

#변수와 상수를 사용해 직선 그리기
a = 1.5 # a: 상수
x = np.linspace(-1,1) # x:변수 -1부터 1의 범위
y = a* x # y: 변수 ---> y=ax=1.5*x의 1차 함수

plt.plot(x,y)
plt.xlabel("x", size = 14)
plt.ylabel("y", size = 14)
plt.grid()
plt.show()


# In[5]:


import numpy as np
import matplotlib.pyplot as plt

b = 3 #b: 상수
x = np.linspace(-1,1)
y = b*x+1  # y=bx+1 ---> y절편이 1인 1차 함수

plt.plot(x,y)
plt.xlabel("x", size=14)
plt.ylabel("y", size=14)
plt.grid()
plt.show()


# In[8]:


#Qeustion. 기존 코드에서 수식 y=4x+1코드를 구현하라.
import numpy as np

def my_func(x):
    return 4*x+1

x = 3
y = my_func(x)
print(y)


# In[5]:


#거듭제곱의 그래프 그리기
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt

def my_func(x):
    a = 3
    return x**a # x의 a제곱


x = np.linspace(0, 2) #0의 3제곱부터 2의 3제곱까지 그린 x축의 포물선
y = my_func(x) # y=f(x)

plt.plot(x,y)
plt.xlabel("x", size = 14)
plt.ylabel("y", size = 14)
plt.grid()
plt.show()


# In[17]:


#제곱근 그래프 그리기
import numpy as np
import matplotlib.pyplot as plt

def my_func(x):
    a=1/2
    return (x**a)+1
    #return np.sqrt(x) + 1 --- 모범답안

x = np.linspace(0,4)
y = my_func(x)

plt.plot(x, y)
plt.xlabel("x", size=14)
plt.ylabel("y", size=14)
plt.grid()
plt.show()


# In[1]:


#2차 다항식 그래프로 그리기
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt

def my_func(x):
    return 3*x**2 + 2*x +1 #f(x)=3x^2 + 2x + 1

x = np.linspace(-2,2)
y = my_func(x)

plt.plot(x,y)
plt.xlabel("x", size=14)
plt.ylabel("y", size=14)
plt.grid()
plt.show()


# In[2]:


#3차 다항식 그래프로 그리기
#y=4x^3 + 2x^2 + x + 3

def my_func(x):
    return 4*x**3 + 2*x**2 + x + 3

x = np.linspace(-2,2)
y = my_func(x)

plt.plot(x, y)
plt.xlabel("x", size=14)
plt.ylabel("y", size=14)
plt.grid()
plt.show()


# In[10]:


#삼각함수 관련
#1.직각삼각형 2.y=sin(x), y=cos(x) 그래프 그리기 3.인수의 단위: 라디안 4.원주율 얻기 위한 호출: np.pi

get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt

def my_sin(x):
    return np.sin(x) #sin(x)

def my_cos(x):
    return np.cos(x) #cos(x)

x = np.linspace(-np.pi,np.pi) # -라디안 ~ +라디안까지
y_sin = my_sin(x)
y_cos = my_cos(x)

plt.plot(x, y_sin, label="sin")
plt.plot(x, y_cos, label="cos")
plt.legend()

plt.xlabel("x", size=14)
plt.ylabel("y", size=14)
plt.grid()

plt.show()


# In[31]:


#삼각함수 관련
#1.직각삼각형 2.y=sin(x), y=cos(x) 그래프 그리기 3.인수의 단위: 라디안 4.원주율 얻기 위한 호출: np.pi

get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt

def my_tan(x):
    return np.tan(x) #tan(x)

x = np.linspace(-np.pi/2,np.pi/2) # -라디안/2 ~ +라디안/2까지
y_tan = my_tan(x)

plt.plot(x, y_tan, label="tan")
plt.legend()

plt.xlabel("x", size=14)
plt.ylabel("y", size=14)
plt.grid()

plt.show()


# In[27]:


import numpy as np
import matplotlib.pyplot as plt

def my_sin(x):
    return np.sin(x)

def my_cos(x):
    return np.cos(x)

x = np.linspace(-2*np.pi, 2*np.pi) #x의 범위를 지정
y_sin = my_sin(x)
y_cos = my_cos(x)

plt.plot(x, y_sin, label="sin")
plt.plot(x, y_cos, label="cos")
plt.legend()

plt.grid()
plt.show()


# In[ ]:




