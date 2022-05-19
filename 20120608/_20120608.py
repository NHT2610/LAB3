import numpy as np
import pandas as pd

MAX_VALUE = 999999999999999
data = pd.read_csv("D:/wine.csv", sep=";")
data

def ls(A, b):
    cb = np.array(b).reshape(len(b), 1)
    C = np.matmul(np.transpose(A), A)
    D = np.matmul(np.transpose(A), cb)
    return np.matmul(np.linalg.inv(C), D)

def norm2(r):
   return sum(a * a for a in r)

# Mô hình tuyến tính: Y = a + b*X
def getA1(X):
    c1 = [1] * len(X)
    K = np.array([c1, X])
    return np.transpose(K)  

def getR1(res, X, Y):
    return [res[0] + res[1] * a - b for a, b in zip(X, Y)]

def F1(data):
    _norm2 = [MAX_VALUE] * 2
    for i in data.keys():
        if i == 'quality':
            break
        X = data[i]
        Y = data.quality
        A = getA1(X)
        res = ls(A, Y)
        r = getR1(res, X, Y)
        print(i + ": ", end = "")
        print("Y = " + str(res[0]) + "+" + str(res[1]) + "*X" + "; ", end = "")
        n2 = norm2(r)
        print("norm2 = " + str(n2))
        if n2 < _norm2[1]:
            _norm2[0] = i
            _norm2[1] = n2   
    print("# Best: ")
    print(str(_norm2[0]) + ": norm2 = " + str(_norm2[1]))

print("$$ Model: Y = a + b*X")
F1(data)

print()

# Mô hình cubic: Y = a + b*X^2
def getA2(X):
    c1 = [1] * len(X)
    c2 = [a * a for a in X]
    K = np.array([c1, c2])
    return np.transpose(K)

def getR2(res, X, Y):
    return [res[0] + res[1] * a * a - b for a, b in zip(X, Y)]

def F2(data):
    _norm2 = [MAX_VALUE] * 2
    for i in data.keys():
        if i == 'quality':
            break
        X = data[i]
        Y = data.quality
        A = getA2(X)
        res = ls(A, Y)
        r = getR2(res, X, Y)
        print(i + ": ", end = "")
        print("Y = " + str(res[0]) + "+" + str(res[1]) + "*X^2" + "; ", end = "")
        n2 = norm2(r)
        print("norm2 = " + str(n2))
        if n2 < _norm2[1]:
            _norm2[0] = i
            _norm2[1] = n2   
    print("# Best: ")
    print(str(_norm2[0]) + ": norm2 = " + str(_norm2[1]))
    
print("$$ Model: Y = a + b*X^2")
F2(data)

print()

# Mô hình đa thức: Y = a + b*X + c*X^2
def getA3(X):
    c1 = [1] * len(X)
    c3 = [a * a for a in X]
    K = np.array([c1, X, c3])
    return np.transpose(K)

def getR3(res, X, Y):
    return [res[0] + res[1] * a + res[2] * a * a - b for a, b in zip(X, Y)]

def F3(data):
    _norm2 = [MAX_VALUE] * 2
    for i in data.keys():
        if i == 'quality':
            break
        X = data[i]
        Y = data.quality
        A = getA3(X)
        res = ls(A, Y)
        r = getR3(res, X, Y)
        print(i + ": ", end = "")
        print("Y = " + str(res[0]) + "+" + str(res[1]) + "*X + " + str(res[2]) + "*X^2" + "; ", end = "")
        n2 = norm2(r)
        print("norm2 = " + str(n2))
        if n2 < _norm2[1]:
            _norm2[0] = i
            _norm2[1] = n2   
    print("# Best: ")
    print(str(_norm2[0]) + ": norm2 = " + str(_norm2[1]))

print("$$ Model: Y = a + b*X + c*X^2")
F3(data)

print()

# Mô hình tuyến tính - log: Y = a + b*lnX 
def getA4(X):
    c1 = [1] * len(X)
    c2 = []
    for a in X:
        if a > 0:
            c2.append(np.log(a))
        else:
            return []
    K = np.array([c1, c2])
    return np.transpose(K)

def getR4(res, X, Y):
    return [res[0] + res[1] * np.log(a) - b for a, b in zip(X, Y)]

def F4(data):
    _norm2 = [MAX_VALUE] * 2
    for i in data.keys():
        if i == 'quality':
            break
        X = data[i]
        Y = data.quality
        A = getA4(X)
        if len(A) == 0:
            print(i + ": " + "Unsatisfactory data!")
            continue
        res = ls(A, Y)
        r = getR4(res, X, Y)
        print(i + ": ", end = "")
        print("logY = " + str(res[0]) + "+" + str(res[1]) + "*lnX" + "; ", end = "")
        n2 = norm2(r)
        print("norm2 = " + str(n2))
        if n2 < _norm2[1]:
            _norm2[0] = i
            _norm2[1] = n2   
    print("# Best: ")
    print(str(_norm2[0]) + ": norm2 = " + str(_norm2[1]))

print("$$ Model: Y = a + b*lnX")
F4(data)

print()

# Mô hình log - tuyến tính: lnY = a + b*X
def getA5(X):
    c1 = [1] * len(X)
    K = np.array([c1, X])
    return np.transpose(K)

def getR5(res, X, Z):
    return [res[0] + res[1] * a - b for a, b in zip(X, Z)]

def F5(data):
    _norm2 = [MAX_VALUE] * 2
    for i in data.keys():
        if i == 'quality':
            break
        X = data[i]
        Y = data.quality
        Z = [np.log(a) for a in Y]
        A = getA5(X)
        res = ls(A, Z)
        r = getR5(res, X, Z)
        print(i + ": ", end = "")
        print("lnY = " + str(res[0]) + "+" + str(res[1]) + "*X" + "; ", end = "")
        n2 = norm2(r)
        print("norm2 = " + str(n2))
        if n2 < _norm2[1]:
            _norm2[0] = i
            _norm2[1] = n2   
    print("# Best: ")
    print(str(_norm2[0]) + ": norm2 = " + str(_norm2[1]))

print("$$ Model: lnY = a + b*X")
F5(data)