import numpy as np
import pandas as pd

def xder(x,theta,y,nill):
    a = (np.dot(theta,np.transpose(x))-y)
    b = a*nill
    xder = np.dot(np.transpose(b),theta)
    return(xder)
    
def thetader(x,theta,y,nill):
    a = (np.dot(theta,np.transpose(x))-y)
    b = a*nill
    thetader = np.dot(b,x)
    return(thetader)
def update(x,theta,y,alpha,iterations,nill):
    for i in range(iterations):
        xder1 = xder(x,theta,y,nill)
        x = x-(alpha*xder1)
        thetader1 = thetader(x,theta,y,nill)
        theta = theta-(alpha*thetader1)
        loss(x,theta,y,nill)
    return(x,theta)
def loss(x,theta,y,nill):
    l = np.sum(((np.dot(theta,x.T)*nill)-y)**2)
    print(l)

def main():
    data = pd.read_csv('matrix1.csv').values
    data = data[:,1:]
    
    mean = pd.read_csv('mean1.csv').values
    mean = np.transpose(mean[:,2:])
  
    y = data-mean 
    ntheta,nx = data.shape
    categories = 18
    iterations = 2000
    learning_rate = 0.03
    nill = (data>0)
    y = y*nill
    theta = np.random.randn(ntheta,categories)*0.1

    
    x = np.random.randn(nx,categories)*0.1
    
    x,theta= update(x,theta,y,learning_rate,iterations,nill)
    ynew = np.dot(theta,np.transpose(x))+mean
    ynew= np.clip(ynew,0,5)
    dk = pd.DataFrame(ynew)
    dk.to_csv('results.csv')
    
    dk1 = pd.DataFrame(x)
    dk1.to_csv('x.csv')
    
    dk2 = pd.DataFrame(theta)
    dk2.to_csv('theta.csv')

    
      
main()
