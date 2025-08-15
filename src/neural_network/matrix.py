"""
Classe Matrix - Implementação de operações básicas de álgebra linear.
Implementação pura em Python sem dependências externas.
"""

import random
import math
from typing import List, Union, Callable


class Matrix:
    """
    Classe para operações de matriz usando apenas Python puro.
    Suporta operações básicas de álgebra linear necessárias para redes neurais.
    """
    
    def __init__(self, data: List[List[float]] = None, rows: int = None, cols: int = None):
        """
        Inicializa uma matriz.
        
        Args:
            data: Lista de listas representando a matriz
            rows: Número de linhas (se data não fornecido)
            cols: Número de colunas (se data não fornecido)
        """
        if data is not None:
            self.data = [row[:] for row in data]  # Cópia profunda
            self.rows = len(data)
            self.cols = len(data[0]) if data else 0
        elif rows is not None and cols is not None:
            self.rows = rows
            self.cols = cols
            self.data = [[0.0 for _ in range(cols)] for _ in range(rows)]
        else:
            raise ValueError("Deve fornecer data ou (rows, cols)")
        
        # Validar consistência
        if self.rows > 0:
            for i, row in enumerate(self.data):
                if len(row) != self.cols:
                    raise ValueError(f"Linha {i} tem {len(row)} elementos, esperado {self.cols}")
    
    def __getitem__(self, key):
        """Permite acesso matrix[i][j] ou matrix[i, j]"""
        if isinstance(key, tuple):
            row, col = key
            return self.data[row][col]
        return self.data[key]
    
    def __setitem__(self, key, value):
        """Permite atribuição matrix[i][j] = value ou matrix[i, j] = value"""
        if isinstance(key, tuple):
            row, col = key
            self.data[row][col] = value
        else:
            self.data[key] = value
    
    def __str__(self):
        """Representação string da matriz"""
        lines = []
        for row in self.data:
            formatted_row = [f"{val:8.4f}" for val in row]
            lines.append("[" + " ".join(formatted_row) + "]")
        return "\n".join(lines)
    
    def __repr__(self):
        return f"Matrix({self.rows}x{self.cols})"
    
    def copy(self) -> 'Matrix':
        """Retorna uma cópia profunda da matriz"""
        return Matrix([row[:] for row in self.data])
    
    def shape(self) -> tuple:
        """Retorna a forma da matriz (rows, cols)"""
        return (self.rows, self.cols)
    
    def fill(self, value: float) -> 'Matrix':
        """Preenche toda a matriz com um valor"""
        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] = value
        return self
    
    def zeros(self) -> 'Matrix':
        """Preenche a matriz com zeros"""
        return self.fill(0.0)
    
    def ones(self) -> 'Matrix':
        """Preenche a matriz com uns"""
        return self.fill(1.0)
    
    def random(self, min_val: float = -1.0, max_val: float = 1.0) -> 'Matrix':
        """Preenche a matriz com valores aleatórios"""
        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] = random.uniform(min_val, max_val)
        return self
    
    def random_normal(self, mean: float = 0.0, std: float = 1.0) -> 'Matrix':
        """Preenche a matriz com valores de distribuição normal"""
        for i in range(self.rows):
            for j in range(self.cols):
                # Box-Muller transform para distribuição normal
                u1 = random.random()
                u2 = random.random()
                z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
                self.data[i][j] = mean + std * z0
        return self
    
    def xavier_init(self) -> 'Matrix':
        """Inicialização Xavier/Glorot"""
        scale = math.sqrt(6.0 / (self.rows + self.cols))
        return self.random(-scale, scale)
    
    def he_init(self) -> 'Matrix':
        """Inicialização He"""
        std = math.sqrt(2.0 / self.rows)
        return self.random_normal(0.0, std)
    
    def add(self, other: Union['Matrix', float]) -> 'Matrix':
        """Soma de matrizes ou escalar"""
        result = Matrix(rows=self.rows, cols=self.cols)
        
        if isinstance(other, Matrix):
            if self.shape() != other.shape():
                raise ValueError(f"Formas incompatíveis: {self.shape()} vs {other.shape()}")
            
            for i in range(self.rows):
                for j in range(self.cols):
                    result.data[i][j] = self.data[i][j] + other.data[i][j]
        else:  # Escalar
            for i in range(self.rows):
                for j in range(self.cols):
                    result.data[i][j] = self.data[i][j] + other
        
        return result
    
    def subtract(self, other: Union['Matrix', float]) -> 'Matrix':
        """Subtração de matrizes ou escalar"""
        result = Matrix(rows=self.rows, cols=self.cols)
        
        if isinstance(other, Matrix):
            if self.shape() != other.shape():
                raise ValueError(f"Formas incompatíveis: {self.shape()} vs {other.shape()}")
            
            for i in range(self.rows):
                for j in range(self.cols):
                    result.data[i][j] = self.data[i][j] - other.data[i][j]
        else:  # Escalar
            for i in range(self.rows):
                for j in range(self.cols):
                    result.data[i][j] = self.data[i][j] - other
        
        return result
    
    def multiply(self, other: Union['Matrix', float]) -> 'Matrix':
        """Multiplicação de matrizes ou elemento-wise com escalar"""
        if isinstance(other, Matrix):
            # Multiplicação de matrizes
            if self.cols != other.rows:
                raise ValueError(f"Dimensões incompatíveis para multiplicação: {self.shape()} x {other.shape()}")
            
            result = Matrix(rows=self.rows, cols=other.cols)
            for i in range(self.rows):
                for j in range(other.cols):
                    sum_val = 0.0
                    for k in range(self.cols):
                        sum_val += self.data[i][k] * other.data[k][j]
                    result.data[i][j] = sum_val
            
            return result
        else:  # Escalar
            result = Matrix(rows=self.rows, cols=self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    result.data[i][j] = self.data[i][j] * other
            return result
    
    def element_wise_multiply(self, other: 'Matrix') -> 'Matrix':
        """Multiplicação elemento a elemento (Hadamard product)"""
        if self.shape() != other.shape():
            raise ValueError(f"Formas incompatíveis: {self.shape()} vs {other.shape()}")
        
        result = Matrix(rows=self.rows, cols=self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] = self.data[i][j] * other.data[i][j]
        
        return result
    
    def transpose(self) -> 'Matrix':
        """Transposição da matriz"""
        result = Matrix(rows=self.cols, cols=self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[j][i] = self.data[i][j]
        return result
    
    def apply(self, func: Callable[[float], float]) -> 'Matrix':
        """Aplica uma função a todos os elementos"""
        result = Matrix(rows=self.rows, cols=self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] = func(self.data[i][j])
        return result
    
    def sum(self) -> float:
        """Soma de todos os elementos"""
        total = 0.0
        for row in self.data:
            for val in row:
                total += val
        return total
    
    def mean(self) -> float:
        """Média de todos os elementos"""
        return self.sum() / (self.rows * self.cols)
    
    def max(self) -> float:
        """Valor máximo da matriz"""
        max_val = float('-inf')
        for row in self.data:
            for val in row:
                max_val = max(max_val, val)
        return max_val
    
    def min(self) -> float:
        """Valor mínimo da matriz"""
        min_val = float('inf')
        for row in self.data:
            for val in row:
                min_val = min(min_val, val)
        return min_val
    
    def get_row(self, index: int) -> List[float]:
        """Retorna uma linha como lista"""
        return self.data[index][:]
    
    def get_column(self, index: int) -> List[float]:
        """Retorna uma coluna como lista"""
        return [self.data[i][index] for i in range(self.rows)]
    
    def set_row(self, index: int, values: List[float]):
        """Define uma linha"""
        if len(values) != self.cols:
            raise ValueError(f"Tamanho da linha {len(values)} não corresponde ao número de colunas {self.cols}")
        self.data[index] = values[:]
    
    def set_column(self, index: int, values: List[float]):
        """Define uma coluna"""
        if len(values) != self.rows:
            raise ValueError(f"Tamanho da coluna {len(values)} não corresponde ao número de linhas {self.rows}")
        for i in range(self.rows):
            self.data[i][index] = values[i]
    
    # Operadores sobrecarregados
    def __add__(self, other):
        return self.add(other)
    
    def __sub__(self, other):
        return self.subtract(other)
    
    def __mul__(self, other):
        return self.multiply(other)
    
    def __rmul__(self, other):
        return self.multiply(other)
    
    def __eq__(self, other):
        if not isinstance(other, Matrix):
            return False
        if self.shape() != other.shape():
            return False
        
        tolerance = 1e-10
        for i in range(self.rows):
            for j in range(self.cols):
                if abs(self.data[i][j] - other.data[i][j]) > tolerance:
                    return False
        return True


# Funções utilitárias para criar matrizes
def zeros(rows: int, cols: int) -> Matrix:
    """Cria uma matriz de zeros"""
    return Matrix(rows=rows, cols=cols).zeros()


def ones(rows: int, cols: int) -> Matrix:
    """Cria uma matriz de uns"""
    return Matrix(rows=rows, cols=cols).ones()


def identity(size: int) -> Matrix:
    """Cria uma matriz identidade"""
    result = zeros(size, size)
    for i in range(size):
        result[i, i] = 1.0
    return result


def random_matrix(rows: int, cols: int, min_val: float = -1.0, max_val: float = 1.0) -> Matrix:
    """Cria uma matriz com valores aleatórios"""
    return Matrix(rows=rows, cols=cols).random(min_val, max_val)


def from_list(data: List[List[float]]) -> Matrix:
    """Cria uma matriz a partir de uma lista de listas"""
    return Matrix(data)


def vector(values: List[float], column: bool = True) -> Matrix:
    """Cria um vetor (matriz coluna ou linha)"""
    if column:
        return Matrix([[val] for val in values])
    else:
        return Matrix([values])
