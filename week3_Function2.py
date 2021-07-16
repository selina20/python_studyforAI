#!/usr/bin/env python
# coding: utf-8

# In[2]:


#총합 = 여러 개의 수치를 모두 더하는 것
#수치의 총 개수 일반화화면 n 개, 총합은 시그마(sum() 함수)로 표현

import numpy as np
import matplotlib.pyplot as plt

a = np.array([1,3,2,5,4]) #a1~a5
y = np.sum(a) # 총합
print(y)


# In[3]:


#총곱 = Numpy의 prod()함수로 구한다

a = np.array([1,3,2,5,4])
y = np.prod(a)
print(y)


# In[6]:


#배열 b의 총합과 총곱 계산

b = np.array([6,1,5,4,3,2])

sum_y = np.sum(b)
prod_y = np.prod(b)

print("총합:", sum_y, "총곱:", prod_y)


# In[31]:


#난수 = 규칙성이 없는 예측할 수 없는 수치
#인공지능에서는 파라미터의 초기화에 난수가 활용됨

#1~6까지의 정수의 난수 생성
import numpy as np

r_int = np.random.randint(6) + 1
print(r_int)


# In[44]:


#0~1까지의 소수의 난수 생성

import numpy as np

r_dec = np.random.rand() #0부터 1사이의 소수를 랜덤으로 반환한다
print(r_dec)


# In[61]:


#난수의 균일한 분포

get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt

n = 500
x = np.random.rand(n) #0~1사이의 n개의 난수가 균일한 확률로 반환됨
y = np.random.rand(n) #0~1사이의 n개의 난수가 균일한 확률로 반환됨

plt.scatter(x,y)
plt.grid()
plt.show()


# In[74]:


#정규분포를 따르는 난수의 분포

#난수가 결정될 확률은 균일하다고 할 수 없음
#Numpy의 random.randn() 함수는 정규분포라는 분포를 따르는 확률로 난수를 반환함

import numpy as np
import matplotlib.pyplot as plt

n = 1000 # 샘플 수
x = np.random.randn(n) #정규분포를 따르는 난수의 분포
y = np.random.randn(n) #정규분포를 따른 난수의 분포

plt.scatter(x,y) #산포도의 플롯
plt.grid()
plt.show()


# In[131]:


#문제)실행할때마다 1~10까지의 정수가 랜덤으로 표시되게 하기

import numpy as np
import matplotlib.pyplot as plt

r_int = np.random.randint(10) + 1
print(r_int)


# #LaTeX(레이텍, 라텍)의 기초
# #LaTeX은 문서 처리 시스템을 사용해 수식을 깨끗하게 입력하는 방법
# 
# #장점: 보기가 좋고 재이용할 수 있는 수식을 손쉽게 이용할 수 있음
# 
# 
# #y=2x+1을 LaTeX를 사용해서 나타내기
# $$y=2x+1$$

# 라텍으로 다양한 수식 나타내기 
# 
# $$a_1$$
# $$a_{{i}}$$
# $$b^2$$
# $$b^{ij}$$
# $$c_1^2$$
# 
# $$y=x^3+2x^2+x+3$$
# $$y=\sqrt x$$
# $$y=\sin x$$
# $$y=\frac{17}{24}$$
# $$y=\sum_{k=1}^n a_k$$

# 문제) LaTeX형식으로 기술하기
# 
# $$y=x^3 + \sqrt x + \frac{a_{ij}}{b_{ij}^4} - \sum_{k=1}^n a_k$$

# In[2]:


#절댓값 (리스트 요소 전체 절댓값)
import numpy as np

x = [-5, 5, -1.28, np.sqrt(5), -np.pi/2]
#여러 값을 리스트에 저장한다

print(np.abs(x)) #절댓값


# In[25]:


#함수의 절댓값
#절댓값의 이미지를 파악하기 위해서 함수의 절댓값을 구해서 그래프로 표시

#삼각함수의 절댓값
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-np.pi, np.pi)
y_sin = np.abs(np.sin(x))
y_cos = np.abs(np.cos(x))
x2 = np.linspace(-1.4, 1.4)
y_tan = np.abs(np.tan(x2))

plt.title("Absolute value graph of trigonometric functions",fontsize=25)
plt.scatter(x, y_sin, label="sin")
plt.scatter(x, y_cos, label="cos")
plt.scatter(x2, y_tan, label="tan")
plt.legend()
#plt.xlim(3,3)
plt.ylim(0,2)

plt.xlabel("x", size=14)
plt.ylabel("y", size=14)
plt.grid()

plt.show()


# In[13]:


#2차 함수의 절댓값을 취한 그래프
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-4, 4)
y = np.abs(x**2 - 4) #2차 함수의 절댓값 취하기

plt.scatter(x, y)

plt.xlabel("x", size=14)
plt.ylabel("y", size=14)
plt.grid()

plt.show()


# In[ ]:




