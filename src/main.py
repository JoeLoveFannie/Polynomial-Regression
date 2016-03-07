import matplotlib.pyplot as plt
import math
import numpy as np
from Regression import Regression

def main():
    reg = Regression()
    reg.set_max_iter = 20000
    reg.set_lr = 0.01
    reg.set_l2_penalty = 0.002
    reg.set_tolerance = 1e-5
    deg=9

    num_sample = 10
    x = np.arange(0,1,1.0/num_sample).reshape(num_sample,1)
    y_list = [math.sin(2*math.pi*e) for e in x] + np.random.normal(0,0.3,num_sample)
    y = np.array(y_list).reshape(num_sample,1)

    theta = np.zeros((deg+1,1))
    theta, loss, repeat = reg.polynomial_fit(x,y,deg)
    z = np.linspace(0,1,100)
    prediction = reg.predict(z)

    fig = plt.figure()
    plt.plot(x,y,'o',label='Input data')
    plt.plot(z,prediction,'r-',label='Prediction')
    plt.plot(z,[math.sin(2*math.pi*e) for e in z], label='Sine Function')
    pylab.xlim([0,1])
    pylab.ylim([-1.5,1.5])
    plt.legend(loc=3)
    fig.suptitle('Polynomial Regression, N=10,Dgree=3,Lamda=0.002')
    plt.xlabel('Input')
    plt.ylabel('Output(prediction)')
    
if __name__ = '__main__':
    main()