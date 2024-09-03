class State:
    """stores a single scalar value and its gradient"""

    def __init__(self, data, _childern=(), _op= ''):
        self.data = data
        self.grad = 0
        #internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_childern)
        self._op = _op #the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, State) else State(other)
        out = State(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, State) else State(other)
        out = State(self.data  * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        out._backward - _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = State(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = State(0 if self.data < 0 else self.data, (self, ), 'RelU')

        def _backward():
            self.grad += (out.data> 0)*out.grad
        out._backward = _backward

