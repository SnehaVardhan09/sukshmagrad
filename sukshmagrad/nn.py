import random
from tensor import State


class Neuron:

    def __init__(self, nin):
        self.w = [State(random.uniform(-1,1)) for _ in range(nin)]
        self.b = State(random.uniform(-1,1))

    def __call__(self, x):
        act = sum((wi* xi for wi,xi  in zip(self.w, x)), self.b )
        out =  act.tanh()

    def parameters(self):
        return self.w + [self.b]

    # def __repr__(self):
    #     return f"{}"

class Layer:

    def __init__(self, nin, nout): #nin = no of inputs, nout = no of neurons
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return out[0] if len(out)==1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:

    def __init(self, nin, nouts):
        sz = [nin] + nout
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    
    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        return x


    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


