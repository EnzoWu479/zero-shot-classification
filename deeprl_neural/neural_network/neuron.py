"""
Implementação da classe Neuron - Neurônio individual.
Implementação pura em Python sem dependências externas.
"""

import random
from typing import List, Union, Optional
from .matrix import Matrix, vector
from .activation import ActivationFunction, get_activation_function


class Neuron:
    """
    Representa um neurônio individual em uma rede neural.
    
    Cada neurônio tem:
    - Pesos (weights) para as conexões de entrada
    - Bias (viés)
    - Função de ativação
    - Valor de saída
    """
    
    def __init__(self, 
                 num_inputs: int,
                 activation: Union[str, ActivationFunction] = 'sigmoid',
                 weight_init_method: str = 'random'):
        """
        Inicializa um neurônio.
        
        Args:
            num_inputs: Número de entradas (conexões) do neurônio
            activation: Função de ativação ('sigmoid', 'relu', 'tanh', etc.) ou instância
            weight_init_method: Método de inicialização dos pesos
        """
        self.num_inputs = num_inputs
        
        # Configurar função de ativação
        if isinstance(activation, str):
            self.activation_func = get_activation_function(activation)
            self.activation_name = activation
        else:
            self.activation_func = activation
            self.activation_name = activation.__class__.__name__.lower()
        
        # Inicializar pesos e bias
        self.weights = Matrix(rows=1, cols=num_inputs)
        self.bias = 0.0
        
        # Inicializar valores
        self._initialize_weights(weight_init_method)
        
        # Valores para forward/backward pass
        self.last_input = None
        self.last_weighted_sum = None
        self.last_output = None
        
        # Gradientes
        self.weight_gradients = Matrix(rows=1, cols=num_inputs).zeros()
        self.bias_gradient = 0.0
    
    def _initialize_weights(self, method: str):
        """Inicializa os pesos do neurônio"""
        if method == 'random':
            self.weights.random(-1.0, 1.0)
            self.bias = random.uniform(-1.0, 1.0)
        elif method == 'xavier':
            self.weights.xavier_init()
            self.bias = random.uniform(-0.1, 0.1)
        elif method == 'he':
            self.weights.he_init()
            self.bias = random.uniform(-0.1, 0.1)
        elif method == 'zeros':
            self.weights.zeros()
            self.bias = 0.0
        else:
            raise ValueError(f"Método de inicialização '{method}' não suportado")
    
    def forward(self, inputs: Union[List[float], Matrix]) -> float:
        """
        Executa o forward pass do neurônio.
        
        Args:
            inputs: Entradas do neurônio (lista ou Matrix)
            
        Returns:
            Saída do neurônio após aplicar a função de ativação
        """
        # Converter inputs para Matrix se necessário
        if isinstance(inputs, list):
            if len(inputs) != self.num_inputs:
                raise ValueError(f"Esperado {self.num_inputs} entradas, recebido {len(inputs)}")
            input_matrix = vector(inputs, column=False)  # Vetor linha
        elif isinstance(inputs, Matrix):
            if inputs.cols != self.num_inputs and inputs.rows != self.num_inputs:
                raise ValueError(f"Dimensão de entrada incompatível: {inputs.shape()}")
            if inputs.rows == 1:
                input_matrix = inputs
            elif inputs.cols == 1:
                input_matrix = inputs.transpose()
            else:
                raise ValueError("Entrada deve ser um vetor")
        else:
            raise TypeError("Entrada deve ser uma lista ou Matrix")
        
        # Salvar entrada para backward pass
        self.last_input = input_matrix.copy()
        
        # Calcular soma ponderada: w * x + b
        weighted_sum_matrix = self.weights * input_matrix.transpose()
        self.last_weighted_sum = weighted_sum_matrix[0, 0] + self.bias
        
        # Aplicar função de ativação
        self.last_output = self.activation_func.forward(self.last_weighted_sum)
        
        return self.last_output
    
    def backward(self, output_gradient: float) -> Matrix:
        """
        Executa o backward pass do neurônio.
        
        Args:
            output_gradient: Gradiente da loss em relação à saída deste neurônio
            
        Returns:
            Gradiente da loss em relação às entradas deste neurônio
        """
        if self.last_input is None or self.last_weighted_sum is None:
            raise ValueError("Forward pass deve ser executado antes do backward pass")
        
        # Calcular gradiente da função de ativação
        activation_gradient = self.activation_func.derivative(self.last_weighted_sum)
        
        # Gradiente local (chain rule)
        local_gradient = output_gradient * activation_gradient
        
        # Calcular gradientes dos pesos: dL/dw = local_gradient * input
        for j in range(self.num_inputs):
            self.weight_gradients[0, j] = local_gradient * self.last_input[0, j]
        
        # Calcular gradiente do bias: dL/db = local_gradient
        self.bias_gradient = local_gradient
        
        # Calcular gradiente das entradas: dL/dx = local_gradient * weights
        input_gradients = Matrix(rows=1, cols=self.num_inputs)
        for j in range(self.num_inputs):
            input_gradients[0, j] = local_gradient * self.weights[0, j]
        
        return input_gradients
    
    def update_weights(self, learning_rate: float):
        """
        Atualiza os pesos usando gradiente descendente.
        
        Args:
            learning_rate: Taxa de aprendizado
        """
        # Atualizar pesos: w = w - lr * dL/dw
        for j in range(self.num_inputs):
            self.weights[0, j] -= learning_rate * self.weight_gradients[0, j]
        
        # Atualizar bias: b = b - lr * dL/db
        self.bias -= learning_rate * self.bias_gradient
    
    def get_weights(self) -> List[float]:
        """Retorna os pesos como lista"""
        return self.weights.get_row(0)
    
    def set_weights(self, weights: List[float]):
        """Define os pesos a partir de uma lista"""
        if len(weights) != self.num_inputs:
            raise ValueError(f"Esperado {self.num_inputs} pesos, recebido {len(weights)}")
        self.weights.set_row(0, weights)
    
    def get_bias(self) -> float:
        """Retorna o bias"""
        return self.bias
    
    def set_bias(self, bias: float):
        """Define o bias"""
        self.bias = bias
    
    def reset_gradients(self):
        """Reseta os gradientes para zero"""
        self.weight_gradients.zeros()
        self.bias_gradient = 0.0
    
    def clone(self) -> 'Neuron':
        """Cria uma cópia do neurônio"""
        new_neuron = Neuron(
            num_inputs=self.num_inputs,
            activation=self.activation_func,
            weight_init_method='zeros'
        )
        
        # Copiar pesos e bias
        new_neuron.weights = self.weights.copy()
        new_neuron.bias = self.bias
        
        return new_neuron
    
    def to_dict(self) -> dict:
        """Serializa o neurônio para dicionário"""
        return {
            'num_inputs': self.num_inputs,
            'activation': self.activation_name,
            'weights': self.get_weights(),
            'bias': self.bias
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Neuron':
        """Deserializa o neurônio a partir de dicionário"""
        neuron = cls(
            num_inputs=data['num_inputs'],
            activation=data['activation'],
            weight_init_method='zeros'
        )
        
        neuron.set_weights(data['weights'])
        neuron.set_bias(data['bias'])
        
        return neuron
    
    def get_activation_output(self) -> Optional[float]:
        """Retorna a última saída calculada"""
        return self.last_output
    
    def get_weighted_sum(self) -> Optional[float]:
        """Retorna a última soma ponderada calculada"""
        return self.last_weighted_sum
    
    def __str__(self) -> str:
        """Representação string do neurônio"""
        weights_str = ", ".join([f"{w:.4f}" for w in self.get_weights()])
        return f"Neuron(inputs={self.num_inputs}, activation={self.activation_name}, weights=[{weights_str}], bias={self.bias:.4f})"
    
    def __repr__(self) -> str:
        return f"Neuron({self.num_inputs}, {self.activation_name})"
    
    def to_dict(self) -> dict:
        """
        Converte o neurônio para um dicionário serializável
        
        Returns:
            Dicionário com os dados do neurônio
        """
        return {
            "num_inputs": self.num_inputs,
            "activation_name": self.activation_name,
            "weights": self.get_weights(),
            "bias": self.bias
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Neuron':
        """
        Cria um neurônio a partir de um dicionário
        
        Args:
            data: Dicionário com os dados do neurônio
            
        Returns:
            Neurônio reconstruído
        """
        neuron = cls(
            num_inputs=data["num_inputs"],
            activation=data["activation_name"],
            weight_init_method='zeros'
        )
        
        # Restaurar pesos e bias
        neuron.set_weights(data["weights"])
        neuron.bias = data["bias"]
        
        return neuron


# Funções utilitárias
def create_neurons(num_neurons: int, 
                  num_inputs: int,
                  activation: str = 'sigmoid',
                  weight_init_method: str = 'random') -> List[Neuron]:
    """
    Cria uma lista de neurônios com as mesmas configurações.
    
    Args:
        num_neurons: Número de neurônios a criar
        num_inputs: Número de entradas por neurônio
        activation: Função de ativação
        weight_init_method: Método de inicialização dos pesos
        
    Returns:
        Lista de neurônios
    """
    return [
        Neuron(num_inputs, activation, weight_init_method)
        for _ in range(num_neurons)
    ]


def clone_neurons(neurons: List[Neuron]) -> List[Neuron]:
    """Clona uma lista de neurônios"""
    return [neuron.clone() for neuron in neurons]
