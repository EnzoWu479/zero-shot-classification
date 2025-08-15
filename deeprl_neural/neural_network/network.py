"""
Implementação da classe NeuralNetwork - Rede Neural Completa.
Implementação pura em Python sem dependências externas.
"""

from typing import List, Union, Dict, Any, Optional, Tuple
import json
import random
import math

from .matrix import Matrix, vector
from .layer import Layer
from .activation import get_activation_function


class NeuralNetwork:
    """
    Rede Neural completa com múltiplas camadas.
    
    Suporta:
    - Forward propagation
    - Backward propagation (backpropagation)
    - Treinamento por gradiente descendente
    - Diferentes funções de loss
    - Múltiplas arquiteturas de rede
    """
    
    def __init__(self, 
                 layer_sizes: List[int],
                 activations: List[str] = None,
                 weight_init_method: str = 'xavier',
                 network_name: str = 'NeuralNetwork'):
        """
        Inicializa a rede neural.
        
        Args:
            layer_sizes: Lista com número de neurônios em cada camada [input, hidden1, hidden2, ..., output]
            activations: Lista com funções de ativação para cada camada (exceto input)
            weight_init_method: Método de inicialização dos pesos
            network_name: Nome da rede
        """
        if len(layer_sizes) < 2:
            raise ValueError("Rede deve ter pelo menos 2 camadas (input e output)")
        
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.input_size = layer_sizes[0]
        self.output_size = layer_sizes[-1]
        self.network_name = network_name
        self.weight_init_method = weight_init_method
        
        # Configurar ativações
        if activations is None:
            # Padrão: sigmoid para camadas ocultas, linear para saída
            activations = ['sigmoid'] * (self.num_layers - 2) + ['linear']
        
        if len(activations) != self.num_layers - 1:
            raise ValueError(f"Esperado {self.num_layers - 1} ativações, recebido {len(activations)}")
        
        self.activations = activations
        
        # Criar camadas
        self.layers = []
        for i in range(1, self.num_layers):  # Começar da primeira camada oculta
            layer = Layer(
                num_neurons=layer_sizes[i],
                num_inputs=layer_sizes[i-1],
                activation=activations[i-1],
                weight_init_method=weight_init_method,
                layer_name=f"Layer_{i}"
            )
            self.layers.append(layer)
        
        # Histórico de treinamento
        self.training_history = {
            'losses': [],
            'accuracies': [],
            'epochs': 0
        }
        
        # Estado do último forward pass
        self.last_prediction = None
        self.last_input = None
    
    def forward(self, inputs: Union[List[float], Matrix]) -> Matrix:
        """
        Executa forward propagation através de toda a rede.
        
        Args:
            inputs: Entradas da rede
            
        Returns:
            Saídas da rede (última camada)
        """
        # Converter entrada para formato padrão
        if isinstance(inputs, list):
            if len(inputs) != self.input_size:
                raise ValueError(f"Esperado {self.input_size} entradas, recebido {len(inputs)}")
            current_input = inputs
        elif isinstance(inputs, Matrix):
            if inputs.cols == self.input_size and inputs.rows == 1:
                current_input = inputs.get_row(0)
            elif inputs.rows == self.input_size and inputs.cols == 1:
                current_input = inputs.get_column(0)
            else:
                raise ValueError(f"Dimensão de entrada incompatível: {inputs.shape()}")
        else:
            raise TypeError("Entrada deve ser uma lista ou Matrix")
        
        # Salvar entrada
        self.last_input = current_input.copy()
        
        # Propagar através das camadas
        for layer in self.layers:
            current_output = layer.forward(current_input)
            current_input = current_output.get_column(0)  # Converter para lista para próxima camada
        
        # A saída final é a saída da última camada
        self.last_prediction = current_output.copy()
        return self.last_prediction.copy()
    
    def predict(self, inputs: Union[List[float], Matrix]) -> List[float]:
        """
        Faz uma predição com a rede (wrapper para forward).
        
        Args:
            inputs: Entradas da rede
            
        Returns:
            Predições como lista
        """
        output = self.forward(inputs)
        return output.get_column(0)
    
    def backward(self, targets: Union[List[float], Matrix], loss_function: str = 'mse') -> float:
        """
        Executa backward propagation através de toda a rede.
        
        Args:
            targets: Valores alvo
            loss_function: Função de loss ('mse', 'cross_entropy')
            
        Returns:
            Valor da loss
        """
        if self.last_prediction is None:
            raise ValueError("Forward pass deve ser executado antes do backward pass")
        
        # Converter targets para lista
        if isinstance(targets, Matrix):
            if targets.cols == 1:
                target_list = targets.get_column(0)
            elif targets.rows == 1:
                target_list = targets.get_row(0)
            else:
                raise ValueError("Target deve ser um vetor")
        elif isinstance(targets, list):
            target_list = targets
        else:
            raise TypeError("Target deve ser uma lista ou Matrix")
        
        if len(target_list) != self.output_size:
            raise ValueError(f"Esperado {self.output_size} targets, recebido {len(target_list)}")
        
        predictions = self.last_prediction.get_column(0)
        
        # Calcular loss e gradientes
        loss_value, output_gradients = self._calculate_loss_and_gradients(
            predictions, target_list, loss_function
        )
        
        # Backward propagation através das camadas (reverso)
        current_gradients = output_gradients
        
        for layer in reversed(self.layers):
            current_gradients = layer.backward(current_gradients)
            current_gradients = current_gradients.get_row(0)  # Converter para lista
        
        return loss_value
    
    def _calculate_loss_and_gradients(self, 
                                    predictions: List[float], 
                                    targets: List[float], 
                                    loss_function: str) -> Tuple[float, List[float]]:
        """Calcula loss e gradientes baseado na função de loss"""
        
        if loss_function == 'mse':
            # Mean Squared Error
            loss = 0.0
            gradients = []
            
            for pred, target in zip(predictions, targets):
                error = pred - target
                loss += error * error
                gradients.append(2.0 * error)  # Derivada de (pred - target)^2
            
            loss /= len(predictions)  # Média
            gradients = [g / len(predictions) for g in gradients]  # Normalizar gradientes
            
            return loss, gradients
        
        elif loss_function == 'cross_entropy':
            # Cross Entropy (assumindo que a última camada é softmax)
            # Implementação simplificada
            epsilon = 1e-15  # Para evitar log(0)
            loss = 0.0
            gradients = []
            
            for pred, target in zip(predictions, targets):
                pred = max(epsilon, min(1 - epsilon, pred))  # Clamp
                loss -= target * math.log(pred)
                gradients.append(-target / pred)
            
            return loss, gradients
        
        elif loss_function == 'mae':
            # Mean Absolute Error
            loss = 0.0
            gradients = []
            
            for pred, target in zip(predictions, targets):
                error = pred - target
                loss += abs(error)
                gradients.append(1.0 if error > 0 else -1.0)
            
            loss /= len(predictions)
            gradients = [g / len(predictions) for g in gradients]
            
            return loss, gradients
        
        else:
            raise ValueError(f"Função de loss '{loss_function}' não suportada")
    
    def update_weights(self, learning_rate: float):
        """Atualiza os pesos de todas as camadas"""
        for layer in self.layers:
            layer.update_weights(learning_rate)
    
    def reset_gradients(self):
        """Reseta gradientes de todas as camadas"""
        for layer in self.layers:
            layer.reset_gradients()
    
    def train_step(self, 
                   inputs: Union[List[float], Matrix], 
                   targets: Union[List[float], Matrix],
                   learning_rate: float = 0.01,
                   loss_function: str = 'mse') -> float:
        """
        Executa um passo de treinamento completo.
        
        Args:
            inputs: Entradas de treinamento
            targets: Valores alvo
            learning_rate: Taxa de aprendizado
            loss_function: Função de loss
            
        Returns:
            Valor da loss
        """
        # Reset gradients
        self.reset_gradients()
        
        # Forward pass
        self.forward(inputs)
        
        # Backward pass
        loss = self.backward(targets, loss_function)
        
        # Update weights
        self.update_weights(learning_rate)
        
        return loss
    
    def train_batch(self,
                   batch_inputs: List[Union[List[float], Matrix]],
                   batch_targets: List[Union[List[float], Matrix]],
                   learning_rate: float = 0.01,
                   loss_function: str = 'mse') -> float:
        """
        Treina em um batch de dados.
        
        Args:
            batch_inputs: Lista de entradas
            batch_targets: Lista de targets
            learning_rate: Taxa de aprendizado
            loss_function: Função de loss
            
        Returns:
            Loss média do batch
        """
        if len(batch_inputs) != len(batch_targets):
            raise ValueError("Número de inputs e targets deve ser igual")
        
        if len(batch_inputs) == 0:
            raise ValueError("Batch não pode estar vazio")
        
        total_loss = 0.0
        batch_size = len(batch_inputs)
        
        # Reset gradients
        self.reset_gradients()
        
        # Acumular gradientes do batch
        for inputs, targets in zip(batch_inputs, batch_targets):
            # Forward pass
            self.forward(inputs)
            
            # Backward pass (acumula gradientes)
            loss = self.backward(targets, loss_function)
            total_loss += loss
        
        # Atualizar pesos com gradientes médios
        self.update_weights(learning_rate)
        
        return total_loss / batch_size
    
    def evaluate(self,
                test_inputs: List[Union[List[float], Matrix]],
                test_targets: List[Union[List[float], Matrix]],
                loss_function: str = 'mse') -> Dict[str, float]:
        """
        Avalia a rede em dados de teste.
        
        Args:
            test_inputs: Entradas de teste
            test_targets: Targets de teste
            loss_function: Função de loss
            
        Returns:
            Dicionário com métricas
        """
        if len(test_inputs) != len(test_targets):
            raise ValueError("Número de inputs e targets deve ser igual")
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = len(test_inputs)
        
        for inputs, targets in zip(test_inputs, test_targets):
            # Predição
            predictions = self.predict(inputs)
            
            # Converter targets para lista se necessário
            if isinstance(targets, Matrix):
                target_list = targets.get_column(0) if targets.cols == 1 else targets.get_row(0)
            else:
                target_list = targets
            
            # Calcular loss
            loss, _ = self._calculate_loss_and_gradients(predictions, target_list, loss_function)
            total_loss += loss
            
            # Calcular acurácia (para classificação)
            if len(predictions) > 1:  # Multi-class
                pred_class = predictions.index(max(predictions))
                true_class = target_list.index(max(target_list))
                if pred_class == true_class:
                    correct_predictions += 1
            else:  # Regressão ou classificação binária
                # Para classificação binária, usar threshold 0.5
                pred_binary = 1 if predictions[0] > 0.5 else 0
                true_binary = 1 if target_list[0] > 0.5 else 0
                if pred_binary == true_binary:
                    correct_predictions += 1
        
        metrics = {
            'loss': total_loss / total_samples,
            'accuracy': correct_predictions / total_samples,
            'total_samples': total_samples
        }
        
        return metrics
    
    def get_network_info(self) -> Dict[str, Any]:
        """Retorna informações sobre a arquitetura da rede"""
        total_params = sum(layer.get_parameter_count() for layer in self.layers)
        
        layer_info = []
        for i, layer in enumerate(self.layers):
            layer_info.append({
                'layer_index': i,
                'layer_name': layer.layer_name,
                'neurons': layer.num_neurons,
                'inputs': layer.num_inputs,
                'activation': layer.activation,
                'parameters': layer.get_parameter_count()
            })
        
        return {
            'network_name': self.network_name,
            'architecture': self.layer_sizes,
            'activations': self.activations,
            'total_parameters': total_params,
            'num_layers': self.num_layers,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'layer_details': layer_info,
            'weight_init_method': self.weight_init_method
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a rede para dicionário"""
        return {
            'network_name': self.network_name,
            'layer_sizes': self.layer_sizes,
            'activations': self.activations,
            'weight_init_method': self.weight_init_method,
            'layers': [layer.to_dict() for layer in self.layers],
            'training_history': self.training_history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NeuralNetwork':
        """Deserializa a rede a partir de dicionário"""
        network = cls(
            layer_sizes=data['layer_sizes'],
            activations=data['activations'],
            weight_init_method=data['weight_init_method'],
            network_name=data['network_name']
        )
        
        # Restaurar camadas
        network.layers = []
        for layer_data in data['layers']:
            layer = Layer.from_dict(layer_data)
            network.layers.append(layer)
        
        # Restaurar histórico
        network.training_history = data.get('training_history', {
            'losses': [],
            'accuracies': [],
            'epochs': 0
        })
        
        return network
    
    def save_to_file(self, filepath: str):
        """Salva a rede em arquivo JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'NeuralNetwork':
        """Carrega a rede de arquivo JSON"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def __str__(self) -> str:
        """Representação string da rede"""
        info = self.get_network_info()
        return f"NeuralNetwork(name={info['network_name']}, architecture={info['architecture']}, params={info['total_parameters']})"
    
    def __repr__(self) -> str:
        return f"NeuralNetwork({self.layer_sizes})"


# Funções utilitárias
def create_simple_network(input_size: int, 
                         hidden_sizes: List[int], 
                         output_size: int,
                         hidden_activation: str = 'relu',
                         output_activation: str = 'linear') -> NeuralNetwork:
    """
    Cria uma rede neural simples com configuração padrão.
    
    Args:
        input_size: Tamanho da entrada
        hidden_sizes: Lista com tamanhos das camadas ocultas
        output_size: Tamanho da saída
        hidden_activation: Ativação das camadas ocultas
        output_activation: Ativação da camada de saída
        
    Returns:
        Rede neural configurada
    """
    layer_sizes = [input_size] + hidden_sizes + [output_size]
    activations = [hidden_activation] * len(hidden_sizes) + [output_activation]
    
    return NeuralNetwork(
        layer_sizes=layer_sizes,
        activations=activations,
        weight_init_method='xavier'
    )