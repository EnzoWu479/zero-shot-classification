"""
Implementação da classe Layer - Camada de neurônios.
Implementação pura em Python sem dependências externas.
"""

from typing import List, Union, Optional, Dict, Any
from .matrix import Matrix, vector, zeros
from .neuron import Neuron, create_neurons
from .activation import get_activation_function


class Layer:
    """
    Representa uma camada de neurônios em uma rede neural.
    
    Uma camada contém múltiplos neurônios que processam as mesmas entradas
    e produzem múltiplas saídas.
    """
    
    def __init__(self,
                 num_neurons: int,
                 num_inputs: int,
                 activation: str = 'sigmoid',
                 weight_init_method: str = 'random',
                 layer_name: str = None):
        """
        Inicializa uma camada de neurônios.
        
        Args:
            num_neurons: Número de neurônios na camada
            num_inputs: Número de entradas para cada neurônio
            activation: Função de ativação para todos os neurônios
            weight_init_method: Método de inicialização dos pesos
            layer_name: Nome opcional da camada
        """
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs
        self.activation = activation
        self.weight_init_method = weight_init_method
        self.layer_name = layer_name or f"Layer_{num_neurons}_{activation}"
        
        # Criar neurônios
        self.neurons = create_neurons(
            num_neurons=num_neurons,
            num_inputs=num_inputs,
            activation=activation,
            weight_init_method=weight_init_method
        )
        
        # Valores para forward/backward pass
        self.last_input = None
        self.last_output = None
        
        # Gradientes acumulados
        self.input_gradients = None
    
    def forward(self, inputs: Union[List[float], Matrix]) -> Matrix:
        """
        Executa o forward pass da camada.
        
        Args:
            inputs: Entradas da camada
            
        Returns:
            Saídas da camada como Matrix (vetor coluna)
        """
        # Converter entrada para formato padrão
        if isinstance(inputs, list):
            if len(inputs) != self.num_inputs:
                raise ValueError(f"Esperado {self.num_inputs} entradas, recebido {len(inputs)}")
            input_matrix = vector(inputs, column=False)  # Vetor linha
        elif isinstance(inputs, Matrix):
            if inputs.cols == self.num_inputs and inputs.rows == 1:
                input_matrix = inputs
            elif inputs.rows == self.num_inputs and inputs.cols == 1:
                input_matrix = inputs.transpose()
            else:
                raise ValueError(f"Dimensão de entrada incompatível: {inputs.shape()}")
        else:
            raise TypeError("Entrada deve ser uma lista ou Matrix")
        
        # Salvar entrada
        self.last_input = input_matrix.copy()
        
        # Calcular saída de cada neurônio
        outputs = []
        for neuron in self.neurons:
            output = neuron.forward(input_matrix)
            outputs.append(output)
        
        # Converter para Matrix (vetor coluna)
        self.last_output = vector(outputs, column=True)
        
        return self.last_output.copy()
    
    def backward(self, output_gradients: Union[List[float], Matrix]) -> Matrix:
        """
        Executa o backward pass da camada.
        
        Args:
            output_gradients: Gradientes da loss em relação às saídas da camada
            
        Returns:
            Gradientes da loss em relação às entradas da camada
        """
        if self.last_input is None:
            raise ValueError("Forward pass deve ser executado antes do backward pass")
        
        # Converter gradientes para lista
        if isinstance(output_gradients, Matrix):
            if output_gradients.cols == 1:
                grad_list = output_gradients.get_column(0)
            elif output_gradients.rows == 1:
                grad_list = output_gradients.get_row(0)
            else:
                raise ValueError("Gradiente deve ser um vetor")
        elif isinstance(output_gradients, list):
            grad_list = output_gradients
        else:
            raise TypeError("Gradientes devem ser uma lista ou Matrix")
        
        if len(grad_list) != self.num_neurons:
            raise ValueError(f"Esperado {self.num_neurons} gradientes, recebido {len(grad_list)}")
        
        # Calcular gradientes de entrada de cada neurônio
        input_gradients_accumulator = zeros(1, self.num_inputs)
        
        for i, neuron in enumerate(self.neurons):
            # Backward pass para este neurônio
            neuron_input_gradients = neuron.backward(grad_list[i])
            
            # Acumular gradientes das entradas
            input_gradients_accumulator = input_gradients_accumulator.add(neuron_input_gradients)
        
        self.input_gradients = input_gradients_accumulator
        return self.input_gradients.copy()
    
    def update_weights(self, learning_rate: float):
        """
        Atualiza os pesos de todos os neurônios da camada.
        
        Args:
            learning_rate: Taxa de aprendizado
        """
        for neuron in self.neurons:
            neuron.update_weights(learning_rate)
    
    def reset_gradients(self):
        """Reseta os gradientes de todos os neurônios"""
        for neuron in self.neurons:
            neuron.reset_gradients()
    
    def get_weights_matrix(self) -> Matrix:
        """
        Retorna uma matriz com todos os pesos da camada.
        Cada linha representa os pesos de um neurônio.
        """
        weight_matrix = Matrix(rows=self.num_neurons, cols=self.num_inputs)
        
        for i, neuron in enumerate(self.neurons):
            weights = neuron.get_weights()
            weight_matrix.set_row(i, weights)
        
        return weight_matrix
    
    def set_weights_matrix(self, weight_matrix: Matrix):
        """
        Define os pesos da camada a partir de uma matriz.
        
        Args:
            weight_matrix: Matrix onde cada linha são os pesos de um neurônio
        """
        if weight_matrix.shape() != (self.num_neurons, self.num_inputs):
            raise ValueError(f"Forma da matriz de pesos incompatível: {weight_matrix.shape()}")
        
        for i, neuron in enumerate(self.neurons):
            weights = weight_matrix.get_row(i)
            neuron.set_weights(weights)
    
    def get_biases(self) -> List[float]:
        """Retorna os biases de todos os neurônios"""
        return [neuron.get_bias() for neuron in self.neurons]
    
    def set_biases(self, biases: List[float]):
        """Define os biases de todos os neurônios"""
        if len(biases) != self.num_neurons:
            raise ValueError(f"Esperado {self.num_neurons} biases, recebido {len(biases)}")
        
        for neuron, bias in zip(self.neurons, biases):
            neuron.set_bias(bias)
    
    def get_outputs(self) -> Optional[Matrix]:
        """Retorna as últimas saídas calculadas"""
        return self.last_output.copy() if self.last_output is not None else None
    
    def get_weighted_sums(self) -> List[float]:
        """Retorna as somas ponderadas de todos os neurônios"""
        return [neuron.get_weighted_sum() for neuron in self.neurons]
    
    def clone(self) -> 'Layer':
        """Cria uma cópia da camada"""
        new_layer = Layer(
            num_neurons=self.num_neurons,
            num_inputs=self.num_inputs,
            activation=self.activation,
            weight_init_method='zeros',
            layer_name=self.layer_name
        )
        
        # Copiar neurônios
        new_layer.neurons = [neuron.clone() for neuron in self.neurons]
        
        return new_layer
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a camada para dicionário"""
        return {
            'layer_name': self.layer_name,
            'num_neurons': self.num_neurons,
            'num_inputs': self.num_inputs,
            'activation': self.activation,
            'weight_init_method': self.weight_init_method,
            'neurons': [neuron.to_dict() for neuron in self.neurons]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Layer':
        """Deserializa a camada a partir de dicionário"""
        layer = cls(
            num_neurons=data['num_neurons'],
            num_inputs=data['num_inputs'],
            activation=data['activation'],
            weight_init_method=data['weight_init_method'],
            layer_name=data['layer_name']
        )
        
        # Restaurar neurônios
        layer.neurons = []
        for neuron_data in data['neurons']:
            neuron = Neuron.from_dict(neuron_data)
            layer.neurons.append(neuron)
        
        return layer
    
    def get_parameter_count(self) -> int:
        """Retorna o número total de parâmetros (pesos + biases) da camada"""
        return self.num_neurons * (self.num_inputs + 1)  # +1 para bias
    
    def apply_dropout(self, dropout_rate: float = 0.5, training: bool = True) -> Matrix:
        """
        Aplica dropout às saídas da camada (simulação simples).
        
        Args:
            dropout_rate: Taxa de dropout (0.0 a 1.0)
            training: Se True, aplica dropout; se False, escala as saídas
            
        Returns:
            Saídas com dropout aplicado
        """
        if self.last_output is None:
            raise ValueError("Forward pass deve ser executado antes do dropout")
        
        if not training:
            # Durante inferência, escala as saídas
            return self.last_output.multiply(1.0 - dropout_rate)
        
        # Durante treinamento, aplica máscara de dropout
        import random
        dropped_output = self.last_output.copy()
        
        for i in range(self.num_neurons):
            if random.random() < dropout_rate:
                dropped_output[i, 0] = 0.0
            else:
                # Escala os neurônios que não foram "dropados"
                dropped_output[i, 0] /= (1.0 - dropout_rate)
        
        return dropped_output
    
    def get_activation_stats(self) -> Dict[str, float]:
        """Retorna estatísticas das ativações da camada"""
        if self.last_output is None:
            return {}
        
        outputs = self.last_output.get_column(0)
        
        return {
            'mean': sum(outputs) / len(outputs),
            'min': min(outputs),
            'max': max(outputs),
            'std': self._calculate_std(outputs)
        }
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calcula desvio padrão"""
        if len(values) <= 1:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def __str__(self) -> str:
        """Representação string da camada"""
        return f"Layer(name={self.layer_name}, neurons={self.num_neurons}, inputs={self.num_inputs}, activation={self.activation})"
    
    def __repr__(self) -> str:
        return f"Layer({self.num_neurons}, {self.num_inputs}, {self.activation})"
    
    def to_dict(self) -> dict:
        """
        Converte a camada para um dicionário serializável
        
        Returns:
            Dicionário com os dados da camada
        """
        return {
            "num_neurons": self.num_neurons,
            "num_inputs": self.num_inputs,
            "activation": self.activation,
            "layer_name": self.layer_name,
            "neurons": [neuron.to_dict() for neuron in self.neurons]
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Layer':
        """
        Cria uma camada a partir de um dicionário
        
        Args:
            data: Dicionário com os dados da camada
            
        Returns:
            Camada reconstruída
        """
        layer = cls(
            num_neurons=data["num_neurons"],
            num_inputs=data["num_inputs"],
            activation=data["activation"],
            layer_name=data["layer_name"]
        )
        
        # Substituir neurônios por versões carregadas
        from ..neural_network.neuron import Neuron
        layer.neurons = [Neuron.from_dict(neuron_data) for neuron_data in data["neurons"]]
        
        return layer


# Funções utilitárias
def create_layers(layer_configs: List[tuple]) -> List[Layer]:
    """
    Cria múltiplas camadas a partir de configurações.
    
    Args:
        layer_configs: Lista de tuplas (num_neurons, activation, layer_name)
        
    Returns:
        Lista de camadas conectadas
    """
    layers = []
    
    for i, config in enumerate(layer_configs):
        if len(config) == 2:
            num_neurons, activation = config
            layer_name = f"Layer_{i+1}"
        elif len(config) == 3:
            num_neurons, activation, layer_name = config
        else:
            raise ValueError("Config deve ser (num_neurons, activation) ou (num_neurons, activation, layer_name)")
        
        # Determinar número de entradas
        if i == 0:
            raise ValueError("Primeira camada precisa especificar num_inputs separadamente")
        else:
            num_inputs = layers[-1].num_neurons
        
        layer = Layer(
            num_neurons=num_neurons,
            num_inputs=num_inputs,
            activation=activation,
            layer_name=layer_name
        )
        
        layers.append(layer)
    
    return layers
