'''
Gradient Descent Optimization

Gradient Descent:
    Gradient descent updates the weights based on the derivative of the current loss function 
    (used as the gradient) and the learning rate.

    Formula:
        W_new = W_old - learning rate * gradient
    Problem:
        1. Gradient Descent (weight update) may slow down when encountering flat region
        2. May encounter a saddle point. (gradient = 0)
        3. May encounter local minimum

    Solution: 
        optimize the method by adjusting the learning rate or the gradient mentioned above. 
        Momentum, AdaGrad, RMSProp, Adam

    Momentum:
        Formula:
            St = β * St-1 + (1 - β) * Gt
        Explain:
            St: The current exponential moving average (EMA) result.
            β: Weight adjustment coefficient — the larger it is, the smoother the data becomes.
               The larger the history EMA is, the smaller the curret gradient weight is
            St-1: The past exponential moving average (EMA) result.
            Gt: current gradient(no need to consider past gradient)
        
        Gradient descant formula(with Momentum):
            W_new = W_old - learning rate * St
    AdaGrad:
        Formula:
            accumulated squared gradients:
                St = St-1 + Gt * Gt
                Explain:
                    St: accumulated squared gradients
                    St: pass accumulated squared gradients
                    Gt: Current gradient
            
            learning rate:
                lr = lr/ (sqt(St) + constant)
                Explain:
                    constant: 1e^-10 prevent denominator become 0
                
            Gradient descant formula:
                W_new = W_old - adjusted learning rate * Gt

        Disadvantage:
            May lead to the learning rate to decrease too early and too aggressively, resulting in 
            a learning rate that becomes too small in the later stages of training, making it 
            difficult for the model to find the optimal solution.

    RMSProp -> can be considered an improvement over AdaGrad, introducing a weighted averaging mechanism.
        Formula: exponentially weighted average of accumulated squared gradients
            accumulated squared gradients:
                St = β * St-1 + (1-β) * Gt * Gt
                Explain:
                    St: accumulated squared gradients
                    St: pass accumulated squared gradients
                    Gt: current gradient
                    β: weighted averaging coefficient
            
            learning rate:
                lr = lr/ (sqt(St) + constant)
                Explain:
                    constant: 1e^-10 prevent denominator become 0
                
            Gradient descant formula:
                W_new = W_old - adjusted learning rate * Gt

        Advantage:
            By introducing the decay coefficient 𝛽, the algorithm controls how much historical gradient information is preserved.
    
    Adam(Adaptive Moment Estimation)
        Idea: 
            optimize both the learning rate and the gradient.
        Formula:
            First Moment: The mean of the gradient
                mt = β1 * Mt-1 + (1- β1) * Gt
                St = β2 * St -1 + (1- β2) * Gt * Gt
            Seconf Moment: The standard deviasion of gradient
                Mt^ = Mt / (1 - β1 ^ t)
                St^ = St / (1 - β1 ^ t)
            Weight update formula:
                W_new = W_old - learning rate / (sqrt(St^) + constant) * Mt^
        Summarize:
            Adam = RMSProp + Momentum
'''
import torch
import torch.nn as nn
import torch.optim as opt

#1. Define function, perform Gradient Descent Optimization -> Momentum
def dm01_momentum():
    #1. initialize weight param
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
    #2. define loss function
    criterion = ((w**2) / 2.0)
    #3. create optimization(function object) -> add Momentum base on SGD
    #param1: parameters to be optimized, param2: learning rate, param3: momentum parameter
    optimizer = opt.SGD(params=[w], lr=0.01, momentum=0.9) #momentum = 0(defult), only consider current gradient
    #4. compute gradient: zero gradient + backpropagation + update parameter
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w: {w}, w.grad: {w.grad}')

    #repeat the steps and update the weight for the second time
    #5.1 define loss function
    criterion = ((w**2) / 2.0)
    #5.2. compute gradient: zero gradient + backpropagation + update parameter
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    #5.3 print
    print(f'w: {w}, w.grad: {w.grad}')


#2. Define function, perform Gradient Descent Optimization -> AdaGrad
def dm02_AdaGrad():
    #1. initialize weight param
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
    #2. define loss function
    criterion = ((w**2) / 2.0)
    #3. create optimization(function object) -> base on AdaGrad
    optimizer = opt.Adagrad(params=[w], lr=0.01) 
    #4. compute gradient: zero gradient + backpropagation + update parameter
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w: {w}, w.grad: {w.grad}')

    #repeat the steps and update the weight for the second time
    #5.1 define loss function
    criterion = ((w**2) / 2.0)
    #5.2. compute gradient: zero gradient + backpropagation + update parameter
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    #5.3 print
    print(f'w: {w}, w.grad: {w.grad}')


#3. Define function, perform Gradient Descent Optimization -> RMSProp
def dm03_RMSProp():
    #1. initialize weight param
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
    #2. define loss function
    criterion = ((w**2) / 2.0)  
    #3. create optimization(function object) -> base on RMSProp
    optimizer = opt.RMSprop(params=[w], lr=0.01, alpha=0.99) 
    #4. compute gradient: zero gradient + backpropagation + update parameter
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w: {w}, w.grad: {w.grad}')

    #repeat the steps and update the weight for the second time
    #5.1 define loss function
    criterion = ((w**2) / 2.0)
    #5.2. compute gradient: zero gradient + backpropagation + update parameter
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    #5.3 print
    print(f'w: {w}, w.grad: {w.grad}')

#4. Define function, perform Gradient Descent Optimization -> Adam
def dm04_Adam():
    #1. initialize weight param
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
    #2. define loss function
    criterion = ((w**2) / 2.0)  
    #3. create optimization(function object) -> base on RMSProp
    optimizer = opt.Adam(params=[w], lr=0.01, betas=(0.9,0.999)) #param 1: β for gradient, param 2: β for learning rate
    #4. compute gradient: zero gradient + backpropagation + update parameter
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w: {w}, w.grad: {w.grad}')

    #repeat the steps and update the weight for the second time
    #5.1 define loss function
    criterion = ((w**2) / 2.0)
    #5.2. compute gradient: zero gradient + backpropagation + update parameter
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    #5.3 print
    print(f'w: {w}, w.grad: {w.grad}')



#5. test
if __name__== '__main__':
    dm01_momentum()
    dm02_AdaGrad()
    dm03_RMSProp()
    dm04_Adam()