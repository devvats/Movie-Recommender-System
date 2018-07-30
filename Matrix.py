import numpy as np 
import pandas as pd

def main():
    
    data1 = pd.read_csv('movies.csv').values
    data2 = pd.read_csv('ratings.csv').values   
    movord = data1[:,0]
    movord = pd.to_numeric(movord,downcast='signed')
    userID = data2[:,0]
    userID = pd.to_numeric(userID,downcast= 'signed')
    movieID = data2[:,1]
    movieID = pd.to_numeric(movieID,downcast= 'signed')
    rating = data2[:,2]
    rating = pd.to_numeric(rating,downcast= 'signed')
    s = np.size(data1,0)  
    dt = np.zeros((9125,672))
    
    print(dt)
    for i in range(100004):
        a = userID[i]
        b = movieID[i]
        d = np.where(movord==b)
        c = rating[i]
        dt[d,a]=c
    data = np.transpose(dt[:,1:])
    dk = pd.DataFrame(data)
    dk.to_csv('matrix.csv')
    
main()
       