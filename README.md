### Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:

To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:

Hardware – PCs
Anaconda – Python 3.7 Installation / Jupyter notebook
Algorithm
Import all the required packages.
Display the output values using graphical representation tools as scatter plot and graph.
predict the values using predict() function.
Display the predicted values and end the program

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by:J.Archana priya 
RegisterNumber:  212221230007
*/
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv("/content/ex1.txt",header = None)
plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of city(10,000s")
plt.ylabel("profit ($10,000")
plt.title("Profit Prediction")

def computeCost(x,y,theta):
  m=len(y)
  h=x.dot(theta)
  square_err=(h-y)**2

  return 1/(2*m) * np.sum(square_err)#returning
  
  
  
  data_n = data.values
m=data_n[:,0].size
x=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(x,y,theta)#call function

def gradientDescent(x,y,theta,alpha,num_iters):
  m=len(y)
  j_history=[]
  for i in range(num_iters):
    preds = x.dot(theta)
    error = np.dot(x.transpose(),(preds -y))
    descent = alpha * 1/m * error
    theta-=descent
    j_history.append(computeCost(x,y,theta))
  return theta,j_history


theta,j_history = gradientDescent(x,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" +"+str(round(theta[1,0],2))+"x1")

plt.plot(j_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function")

def predict(x,theta):
  pred = np.dot(theta.transpose(),x)
  return pred[0]

pred1 = predict(np.array([1,3.5]),theta)*10000
print("Population = 35000 , we predict a profit of $"+str(round(pred1,0)))

pred2 = predict(np.array([1,7]),theta)*10000
print("Population = 70000 , we predict a profit of $"+str(round(pred2,0)))
```

## Output:

![image](https://user-images.githubusercontent.com/118706984/233589278-59e8ced1-404e-4d0c-986a-462f3067f4c1.png)

![image](https://user-images.githubusercontent.com/118706984/233589566-718fa5be-77bf-46c3-89ba-da9b82b5b465.png)

![image](https://user-images.githubusercontent.com/118706984/233589619-d3faf218-4d91-47c8-974d-5a017d9d63cf.png)

![image](https://user-images.githubusercontent.com/118706984/233589414-0339dab0-671c-4cbd-9cbf-f62828a6d36f.png)

![image](https://user-images.githubusercontent.com/118706984/233589444-151eff33-b9d1-464c-a351-ef70de8e36f7.png)

![image](https://user-images.githubusercontent.com/118706984/233589786-1b26c883-3c41-4a78-bdfd-322e4a92eade.png)

![image](https://user-images.githubusercontent.com/118706984/233589986-26053536-775e-4555-9cb6-2e1a9c04de6e.png)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
