import torch
import os
import pandas as pd
import numpy as np
import matplotlib_inline as backend_inline
from matplotlib import pyplot as plt
from torch.distributions.multinomial import Multinomial


x = torch.arange(12, dtype = torch.float32)
print(x)
print(x.numel())

X = x.reshape(3,4)
print(X)

Z = torch.zeros((2,3,4))
O = torch.ones((2,3,4))
G = torch.randn(3,4)
M = torch.tensor([[2,2,4,3],[1,2,3,3],[4,3,2,1]])

E = torch.exp(x)

print(E)

os.makedirs(os.path.join('.','data'), exist_ok = True)
data_file = os.path.join('.','data','house_tiny.csv')
print(data_file)
#with open(data_file, 'w') as f:
#    f.write('''NumRooms,RoofType,Price
#NA,NA,127500
#2,NA,106000
#4,Slate,178100
#NA,NA,140000
#    ''')

data = pd.read_csv(data_file)
print(data)

input, target = data.iloc[:,:2], data.iloc[:, 2]

print(input)
#print(target)

input = pd.get_dummies(input, dummy_na = True)

input = input.fillna(input.mean())

print(input)

X = torch.tensor(input.to_numpy(dtype=float))
y = torch.tensor(target.to_numpy(dtype=float))

print(X)
print(y)

def f(x):
    return 3 * x**2 - 4 * x

for h in 10.0**np.arange(-1,-6,-1):
    print(f'h={h:.5f}, numerical limit = {(f(1+h)-f(1))/h:.5f}')

#def use_svg_display():
    #backend_inline.set_matplotlib_formats('svg')


def set_figsize(figsize = (3.5, 2.5)):
    #use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim), axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)

    axes.grid()

def plot(X, Y=None, xlabel=None, ylabel=None, legend=[], xlim=None,
         ylim=None, xscale='linear', yscale='linear', fmts=('-','m--','g-.','r:'), figsize=(3.5,2.5), axes=None):

    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X,list) and not hasattr(X[0], "__len__"))

    if has_one_axis(X): X = [X]
    if Y is None:
        X, Y = [[]]*len(X), X
    elif has_one_axis(Y): Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    set_figsize(figsize)
    if axes is None:
        axes = plt.gca()
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        axes.plot(x,y,fmt) if len(x) else axes.plot(y,fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
    plt.savefig("plot.svg", format="svg")
    plt.show()

#x = np.arange(0,3,0.1)
#plot(x,[f(x), 2*x -3], 'x', 'f(x)',legend=['f(x)','Tangent line(x=1)'])
x = torch.arange(4.0)
x.requires_grad_(True)
y = 2 * torch.dot(x,x)
y.backward()
print(x.grad)
x.grad.zero_()
y = x.sum()
y = x * x
u = y.detach()
print(u)
z = u * x   #[0, 1, 4, 9]
#z = y * x  #[0, 3, 12, 27]

z.sum().backward()
print(x.grad)

def f(a):
    b = a * 2
    while b.norm() < 1000:
        print(f'b.norm: {b.norm()}')
        b = b * 2
        print(f'loop b: {b}')
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True)

print(f'a: {a}')

d = f(a)

print(f'd: {d}')

d.backward()

print(f'a.grad: {a.grad}')