# Doriel Fay 208770289

import numpy as np
import sys

_trainX,_trainY,_testX,_output=sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]

trainX=np.loadtxt(_trainX, dtype=float, delimiter=',')
trainY=np.loadtxt(_trainY, dtype=int)
testX=np.loadtxt(_testX, dtype=float, delimiter=',')
f = open(_output,"w+")
class Knn:
    def __init__(self,k,trainx, trainy):
        self.k = k
        self.z=[]
        for i in range(len(trainx)):
           self.z.append([trainx[i],trainy[i]])

    def givenEx (self,ex):
        distance=[]
        for data in self.z:
            distance.append([np.linalg.norm(ex-data[0]),data[1]])
        distance.sort(key = lambda elem: elem[0])
        counter0=0
        counter1=0
        counter2=0
        for i in range(self.k):
            if distance[i][1]==0:
                counter0+=1
            elif distance[i][1]==1:
                counter1+=1
            else:
                counter2+=1
        if counter0>=counter1 and counter0>=counter2:
            return 0
        elif counter1>=counter0 and counter1>=counter2:
            return 1
        else :
            return 2

    def predict (self,testx):
        pLable=[]
        for ex in testx:
            pLable.append(self.givenEx(ex))
        return np.array(pLable)

class Perceptron:
    def __init__(self,learningRate,maxIteration):
        self.Lr=learningRate
        self.maxIter=maxIteration
    def fit(self,x,y):
        #how many example,and features per ex
        countExample,countFeatures=x.shape
        x = np.insert(x, 0, np.ones(x.shape[0]), axis=1)
        self.w=np.zeros((3,countFeatures+1))
        for iter in range(self.maxIter):
            for i in range(countExample):
                yHat=self.predict_one(x[i])
                if(yHat!=y[i]):
                    self.w[int(y[i])]+=self.Lr*x[i]
                    self.w[yHat]-=self.Lr*x[i]
            no_eroor=sum(np.array([np.argmax(np.dot(self.w,ex)) for ex in x]))
            if no_eroor==0:
                break

    def predict_one(self,x):
        return np.argmax(np.dot(self.w,x))
    def predict(self,X):
        X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
        return np.array([self.predict_one(x) for x in X])

class Svm :
    def __init__(self,maxIteration,learningRate,lamda):
        self.maxIter=maxIteration
        self.lr=learningRate
        self.lamda=lamda
    def fit(self,x,y):
        countExample, countFeatures = x.shape
        self.w = np.zeros((3, countFeatures+1))
        x = np.insert(x, 0, np.ones(x.shape[0]), axis=1)
        for iter in range(self.maxIter):
            for i in range(countExample):
                result=np.dot(self.w,x[i])
                result[int(y[i])]= -np.inf
                yHat=np.argmax(result)
                loss=max(0, 1 - np.dot(self.w[int(y[i])],x[i]) + np.dot(self.w[yHat], x[i]))
                llr=self.lamda*self.lr
                self.w = np.dot(self.w ,(1 - self.lamda * self.lr))
                if loss > 0 :
                    self.w[int(y[i])]+= self.lr*x[i]
                    self.w[yHat]-=self.lr*x[i]
    def prediction(self,X):
        X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
        return np.array([np.argmax(np.dot(self.w,x)) for x in X])

class PassiveAggresive:
    def __init__(self,maxIteration):
        self.maxIter=maxIteration
    def fit(self,x,y):
        countExample, countFeatures = x.shape
        self.w = np.zeros((3, countFeatures+1))
        x = np.insert(x, 0, np.ones(x.shape[0]), axis=1)
        for iter in range(self.maxIter):
            for i in range(countExample):
                yHat = np.argmax(np.dot(self.w,x[i]))
                if yHat != y[i]:
                    wxy=np.dot(self.w[int(y[i])],x[i])+np.dot(self.w[int(yHat)],x[i])
                    loss = max(0,1-wxy)
                    tau = loss/(2*pow(np.linalg.norm(x),2))
                    self.w[int(y[i])] += tau * x[i]
                    self.w[yHat] -= tau * x[i]
    def prediction(self,X):
        X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
        return np.array([np.argmax(np.dot(self.w,x)) for x in X])

#main
knn=Knn(9,trainX,trainY)
predictKnn=knn.predict(testX)
trainX = (trainX - trainX.mean(0)) / trainX.std(0)
testX = (testX - testX.mean(0)) / testX.std(0)
perceprton=Perceptron(0.04, 455)
perceprton.fit(trainX,trainY)
predictPerceprton=perceprton.predict(testX)
svm=Svm(7,0.05,0.003)
svm.fit(trainX,trainY)
predictSvm=svm.prediction(testX)
pa=PassiveAggresive(7)
pa.fit(trainX,trainY)
predictPa=pa.prediction(testX)
for k, p, s, a in zip(predictKnn, predictPerceprton, predictSvm, predictPa):
    f.write(f"knn: {k}, perceptron: {p}, svm: {s}, pa: {a}\n")
f.close()




