"""
Funções de ativação e suas derivadas para redes neurais.
Implementação pura em Python sem dependências externas.
"""

import math
from typing import Union, List
from .matrix import Matrix


class ActivationFunction:
    """Classe base para funções de ativação"""
    
    def forward(self, x: Union[float, Matrix]) -> Union[float, Matrix]:
        """Aplica a função de ativação"""
        raise NotImplementedError
    
    def derivative(self, x: Union[float, Matrix]) -> Union[float, Matrix]:
        """Calcula a derivada da função de ativação"""
        raise NotImplementedError
    
    def __call__(self, x: Union[float, Matrix]) -> Union[float, Matrix]:
        return self.forward(x)


class Sigmoid(ActivationFunction):
    """Função de ativação Sigmoid: f(x) = 1 / (1 + e^(-x))"""
    
    def forward(self, x: Union[float, Matrix]) -> Union[float, Matrix]:
        if isinstance(x, Matrix):
            def sigmoid_func(val):
                # Clamp para evitar overflow
                val = max(-500, min(500, val))
                return 1.0 / (1.0 + math.exp(-val))
            return x.apply(sigmoid_func)
        else:
            # Clamp para evitar overflow
            x = max(-500, min(500, x))
            return 1.0 / (1.0 + math.exp(-x))
    
    def derivative(self, x: Union[float, Matrix]) -> Union[float, Matrix]:
        if isinstance(x, Matrix):
            def sigmoid_derivative(val):
                val = max(-500, min(500, val))
                s = 1.0 / (1.0 + math.exp(-val))
                return s * (1.0 - s)
            return x.apply(sigmoid_derivative)
        else:
            x = max(-500, min(500, x))
            s = 1.0 / (1.0 + math.exp(-x))
            return s * (1.0 - s)


class ReLU(ActivationFunction):
    """Função de ativação ReLU: f(x) = max(0, x)"""
    
    def forward(self, x: Union[float, Matrix]) -> Union[float, Matrix]:
        if isinstance(x, Matrix):
            return x.apply(lambda val: max(0.0, val))
        else:
            return max(0.0, x)
    
    def derivative(self, x: Union[float, Matrix]) -> Union[float, Matrix]:
        if isinstance(x, Matrix):
            return x.apply(lambda val: 1.0 if val > 0 else 0.0)
        else:
            return 1.0 if x > 0 else 0.0


class LeakyReLU(ActivationFunction):
    """Função de ativação Leaky ReLU: f(x) = max(alpha*x, x)"""
    
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
    
    def forward(self, x: Union[float, Matrix]) -> Union[float, Matrix]:
        if isinstance(x, Matrix):
            return x.apply(lambda val: val if val > 0 else self.alpha * val)
        else:
            return x if x > 0 else self.alpha * x
    
    def derivative(self, x: Union[float, Matrix]) -> Union[float, Matrix]:
        if isinstance(x, Matrix):
            return x.apply(lambda val: 1.0 if val > 0 else self.alpha)
        else:
            return 1.0 if x > 0 else self.alpha


class Tanh(ActivationFunction):
    """Função de ativação Tanh: f(x) = tanh(x)"""
    
    def forward(self, x: Union[float, Matrix]) -> Union[float, Matrix]:
        if isinstance(x, Matrix):
            def tanh_func(val):
                # Clamp para evitar overflow
                val = max(-500, min(500, val))
                return math.tanh(val)
            return x.apply(tanh_func)
        else:
            x = max(-500, min(500, x))
            return math.tanh(x)
    
    def derivative(self, x: Union[float, Matrix]) -> Union[float, Matrix]:
        if isinstance(x, Matrix):
            def tanh_derivative(val):
                val = max(-500, min(500, val))
                t = math.tanh(val)
                return 1.0 - t * t
            return x.apply(tanh_derivative)
        else:
            x = max(-500, min(500, x))
            t = math.tanh(x)
            return 1.0 - t * t


class Linear(ActivationFunction):
    """Função de ativação Linear: f(x) = x"""
    
    def forward(self, x: Union[float, Matrix]) -> Union[float, Matrix]:
        return x
    
    def derivative(self, x: Union[float, Matrix]) -> Union[float, Matrix]:
        if isinstance(x, Matrix):
            return x.apply(lambda val: 1.0)
        else:
            return 1.0


class Softmax(ActivationFunction):
    """Função de ativação Softmax: f(x_i) = e^x_i / sum(e^x_j)"""
    
    def forward(self, x: Union[float, Matrix]) -> Union[float, Matrix]:
        if isinstance(x, Matrix):
            if x.rows == 1:
                # Vetor linha
                values = x.get_row(0)
                return self._softmax_vector(values, row=True)
            elif x.cols == 1:
                # Vetor coluna
                values = x.get_column(0)
                return self._softmax_vector(values, row=False)
            else:
                # Aplica softmax em cada linha
                result = Matrix(rows=x.rows, cols=x.cols)
                for i in range(x.rows):
                    row_values = x.get_row(i)
                    softmax_row = self._softmax_list(row_values)
                    result.set_row(i, softmax_row)
                return result
        else:
            # Para um único valor, retorna sigmoid
            x = max(-500, min(500, x))
            return 1.0 / (1.0 + math.exp(-x))
    
    def _softmax_list(self, values: List[float]) -> List[float]:
        """Aplica softmax a uma lista de valores"""
        # Subtract max for numerical stability
        max_val = max(values)
        exp_values = [math.exp(val - max_val) for val in values]
        sum_exp = sum(exp_values)
        
        if sum_exp == 0:
            # Fallback para evitar divisão por zero
            return [1.0 / len(values)] * len(values)
        
        return [exp_val / sum_exp for exp_val in exp_values]
    
    def _softmax_vector(self, values: List[float], row: bool = True) -> Matrix:
        """Converte softmax de lista para Matrix"""
        softmax_values = self._softmax_list(values)
        if row:
            return Matrix([softmax_values])
        else:
            return Matrix([[val] for val in softmax_values])
    
    def derivative(self, x: Union[float, Matrix]) -> Union[float, Matrix]:
        # Para softmax, a derivada é complexa e geralmente calculada
        # junto com a loss function (cross-entropy)
        # Por simplicidade, retornamos a derivada de sigmoid para casos escalares
        if isinstance(x, Matrix):
            # Para matrizes, retorna uma aproximação
            softmax_output = self.forward(x)
            return softmax_output.apply(lambda val: val * (1.0 - val))
        else:
            # Para escalares, usa derivada de sigmoid
            x = max(-500, min(500, x))
            s = 1.0 / (1.0 + math.exp(-x))
            return s * (1.0 - s)


class ELU(ActivationFunction):
    """Função de ativação ELU: f(x) = x if x > 0 else alpha * (e^x - 1)"""
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def forward(self, x: Union[float, Matrix]) -> Union[float, Matrix]:
        if isinstance(x, Matrix):
            def elu_func(val):
                if val > 0:
                    return val
                else:
                    val = max(-500, min(500, val))
                    return self.alpha * (math.exp(val) - 1.0)
            return x.apply(elu_func)
        else:
            if x > 0:
                return x
            else:
                x = max(-500, min(500, x))
                return self.alpha * (math.exp(x) - 1.0)
    
    def derivative(self, x: Union[float, Matrix]) -> Union[float, Matrix]:
        if isinstance(x, Matrix):
            def elu_derivative(val):
                if val > 0:
                    return 1.0
                else:
                    val = max(-500, min(500, val))
                    return self.alpha * math.exp(val)
            return x.apply(elu_derivative)
        else:
            if x > 0:
                return 1.0
            else:
                x = max(-500, min(500, x))
                return self.alpha * math.exp(x)


# Factory function para criar funções de ativação
def get_activation_function(name: str, **kwargs) -> ActivationFunction:
    """
    Factory function para criar funções de ativação.
    
    Args:
        name: Nome da função ('sigmoid', 'relu', 'tanh', 'linear', 'softmax', 'leaky_relu', 'elu')
        **kwargs: Parâmetros específicos da função
    
    Returns:
        Instância da função de ativação
    """
    name = name.lower()
    
    if name == 'sigmoid':
        return Sigmoid()
    elif name == 'relu':
        return ReLU()
    elif name == 'leaky_relu':
        alpha = kwargs.get('alpha', 0.01)
        return LeakyReLU(alpha)
    elif name == 'tanh':
        return Tanh()
    elif name == 'linear':
        return Linear()
    elif name == 'softmax':
        return Softmax()
    elif name == 'elu':
        alpha = kwargs.get('alpha', 1.0)
        return ELU(alpha)
    else:
        raise ValueError(f"Função de ativação '{name}' não suportada")


# Aliases para compatibilidade
sigmoid = Sigmoid()
relu = ReLU()
tanh = Tanh()
linear = Linear()
softmax = Softmax()
leaky_relu = LeakyReLU()
elu = ELU()


# Funções auxiliares para aplicar ativações diretamente
def apply_activation(x: Union[float, Matrix], activation: str, **kwargs) -> Union[float, Matrix]:
    """Aplica uma função de ativação aos dados"""
    func = get_activation_function(activation, **kwargs)
    return func.forward(x)


def apply_activation_derivative(x: Union[float, Matrix], activation: str, **kwargs) -> Union[float, Matrix]:
    """Aplica a derivada de uma função de ativação aos dados"""
    func = get_activation_function(activation, **kwargs)
    return func.derivative(x)
