import numpy as np
import copy
import random


class DNN :
    def __init__(self,params):
        self.params=params
 
    def sigmoid(self , x , derivative = False) : 
        if derivative :
            return self.sigmoid(x , False)*(1 - self.sigmoid(x , False))
        else :
             return 1./(1 + np.exp(-x)) 
        

    def softmax(self, x, derivative=False):
        exps = np.exp(x - x.max())
        softmax_vals = exps / np.sum(exps, axis=0)
        if derivative:
            return softmax_vals * (1 - softmax_vals)
        else:
            return softmax_vals
    
        
    def forward_pass(self, x_train):
        params = self.params
        params['A0'] = x_train

        params['Z1'] = np.dot( params['A0'] , params['W1'].T) + params['b1']
        params['A1'] = self.sigmoid(params['Z1'])

        params['Z2'] = np.dot(params['A1'] , params['W2'].T) + params['b2']
        params['A2'] = self.sigmoid(params['Z2'])

        params['Z3'] = np.dot(params['A2'] , params['W3'].T) + params['b3']
        params['A3'] = self.softmax(params['Z3'])

        return params['A3']
    
    
    def compute_accuracy(self , test_Data) :
        prediction = []
        for x in test_Data : 
            values = x.split(",")
            input_data = (np.asfarray(values[1:])/255.0 * 0.99)+0.01
            targets =  np.zeros(10) +0.01
            targets[int(values[0])] = 0.99
            output = self.forward_pass(input_data)
            pred =  np.argmax(output)
            prediction.append(pred == np.argmax(targets))
        return np.mean(prediction)

    def mse(self ,y_batch , output) : 
        return np.mean((y_batch - output)**2)


    def train(self , train_list) : 
        np.random.shuffle(train_list)

        X_batch = []
        y_batch = []
        for x in train_list[:700] : 
            values = x.split(",")
            input_data = (np.asfarray(values[1:])/255.0 * 0.99)+0.01  
            X_batch.append(input_data)
            targets = np.zeros(10) + 0.01
            targets[int(values[0])] = 0.99
            y_batch.append(targets)

        X_batch = np.array(X_batch)
        output = self.forward_pass(X_batch) 
        y_batch= np.array(y_batch)  

        return self.mse(y_batch , output)    
        


class Wolf:
    def __init__(self,sizes=[784 , 128 , 64 , 10]  , train_list=None , test_list=None) :
        self.params={
            'W1' : np.random.randn(sizes[1] , sizes[0]) * np.sqrt(1./sizes[1]),
            'W2' : np.random.randn(sizes[2] , sizes[1]) * np.sqrt(1./sizes[2]),
            'W3' : np.random.randn(sizes[3] , sizes[2]) * np.sqrt(1./sizes[3]),
            'b1' : np.random.randn(sizes[1] ) * np.sqrt(1./sizes[1]),
            'b2' : np.random.randn(sizes[2] ) * np.sqrt(1./sizes[2]),
            'b3' : np.random.randn(sizes[3]) * np.sqrt(1./sizes[3])
        }
        
        dnn = DNN(self.params)
        self.fitness = dnn.train(train_list)
        



def gwo(maxIter , nbWolves ,train_list ):
    population = [Wolf(train_list=train_list) for i in range(nbWolves)]
    population = sorted(population , key= lambda temp : temp.fitness) 
    alpha_w , beta_w , gamma_w = copy.copy(population[:3])

    iter =0 
    while iter <maxIter :

        if iter % 10 == 0 :
            print("iteration = "+str(iter)+" best fitness = %f"% alpha_w.fitness)


        a = 2*(1- iter/maxIter)

        for i in range(nbWolves) :
            A1 , A2 , A3 = a * (2 * random.random()-1) , a * (2 * random.random()-1) , a * (2 * random.random()-1)
            C1 , C2 , C3 = 2* random.random() , 2* random.random() , 2* random.random()

            X1 ,X2 ,X3 , newX = {} ,{} ,{} , {}
            wolf = Wolf(train_list=train_list)
            for key , values  in wolf.params.items() : 
                X1[key] = alpha_w.params[key] - A1 * abs(C1 * alpha_w.params[key] - population[i].params[key])
                X2[key] = beta_w.params[key] - A2 * abs(C2 * beta_w.params[key] - population[i].params[key])
                X3[key] = gamma_w.params[key] - A3 * abs(C3 * gamma_w.params[key] - population[i].params[key])
                newX[key] = (X1[key] + X2[key] + X3[key])/3.0
            
            dn2 = DNN(newX)
            newFitness = dn2.train(train_list)
            if newFitness < population[i].fitness:
                population[i].params = newX
                population[i].fitness = newFitness

        
        population = sorted(population , key=lambda temp : temp.fitness) 
        alpha_w , beta_w , gamma_w = copy.copy(population[:3])
        iter +=1

    return alpha_w 


if __name__ == "__main__":
    maxIter =100
    nbWolves =20
    train_file = open("mnist_train.csv" , "r")
    train_list = train_file.readlines()
    train_file.close()
    train_file = open("mnist_test.csv" , "r")
    test_list = train_file.readlines()
    train_file.close()
    print("Démarrage Gray Wolf algorithme \n")

    alpha_w = gwo(maxIter , nbWolves , train_list)

    print("Meilleure fitness est " + str(alpha_w.fitness))

    dnn = DNN(params=alpha_w.params)
    print("accurcy de Mrilleure solution est " + str(dnn.compute_accuracy(test_list)))
