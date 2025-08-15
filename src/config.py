"""
Arquivo de configuração para o projeto de Rede Neural Deep Learning com Aprendizado por Reforço.
Este arquivo contém todos os hiperparâmetros e configurações do projeto.
"""

# =============================================================================
# CONFIGURAÇÕES DA REDE NEURAL
# =============================================================================

# Arquitetura da rede
HIDDEN_LAYERS = [64, 64]  # Lista com número de neurônios em cada camada oculta
INPUT_SIZE = None  # Será definido baseado no ambiente
OUTPUT_SIZE = None  # Será definido baseado no número de ações

# Funções de ativação disponíveis: "sigmoid", "relu", "tanh"
ACTIVATION_FUNCTION = "relu"
OUTPUT_ACTIVATION = "linear"  # Para Q-values

# Inicialização de pesos
WEIGHT_INIT_METHOD = "xavier"  # "xavier", "he", "random"
WEIGHT_INIT_SCALE = 0.1

# =============================================================================
# CONFIGURAÇÕES DE APRENDIZADO POR REFORÇO
# =============================================================================

# Hiperparâmetros principais
LEARNING_RATE = 0.001
GAMMA = 0.99  # Discount factor
TAU = 0.001   # Para soft update da target network

# Política de exploração (Epsilon-Greedy)
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
EPSILON_DECAY_TYPE = "exponential"  # "exponential" ou "linear"

# Experience Replay
BUFFER_SIZE = 10000
BATCH_SIZE = 32
MIN_EXPERIENCES = 1000  # Experiências mínimas antes de começar a treinar

# Frequência de atualização
TARGET_UPDATE_FREQUENCY = 100  # Episódios entre atualizações da target network
TRAINING_FREQUENCY = 4  # Passos entre sessões de treinamento

# =============================================================================
# CONFIGURAÇÕES DE TREINAMENTO
# =============================================================================

# Episódios e passos
MAX_EPISODES = 2000
MAX_STEPS_PER_EPISODE = 500
EVALUATION_FREQUENCY = 100  # Episódios entre avaliações

# Critérios de parada antecipada
EARLY_STOPPING = True
PATIENCE = 200  # Episódios sem melhoria antes de parar
MIN_IMPROVEMENT = 0.01  # Melhoria mínima considerada significativa

# =============================================================================
# CONFIGURAÇÕES DE AMBIENTE
# =============================================================================

# GridWorld
GRIDWORLD_SIZE = 5
GRIDWORLD_OBSTACLES = [(2, 2), (3, 1)]
GRIDWORLD_GOAL = (4, 4)
GRIDWORLD_START = (0, 0)

# CartPole (simulado)
CARTPOLE_GRAVITY = 9.8
CARTPOLE_CART_MASS = 1.0
CARTPOLE_POLE_MASS = 0.1
CARTPOLE_POLE_LENGTH = 0.5

# =============================================================================
# CONFIGURAÇÕES DE PERSISTÊNCIA
# =============================================================================

# Formato de salvamento
SAVE_FORMAT = "json"  # "json" ou "pickle"
MODEL_VERSION = "1.0"

# Caminhos
MODELS_DIR = "models"
LOGS_DIR = "logs"
CHECKPOINTS_DIR = "models/checkpoints"

# Frequência de salvamento
SAVE_FREQUENCY = 100  # Episódios entre salvamentos automáticos
KEEP_LAST_N_MODELS = 5  # Número de modelos a manter

# =============================================================================
# CONFIGURAÇÕES DE LOGGING
# =============================================================================

# Nível de log
LOG_LEVEL = "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR"

# Métricas a registrar
LOG_REWARDS = True
LOG_LOSS = True
LOG_EPSILON = True
LOG_Q_VALUES = False  # Pode gerar muito log

# Formato de saída
LOG_TO_FILE = True
LOG_TO_CONSOLE = True
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# =============================================================================
# CONFIGURAÇÕES DE DEBUGGING E VALIDAÇÃO
# =============================================================================

# Modo debug
DEBUG_MODE = False
VERBOSE_TRAINING = False

# Validação
VALIDATE_GRADIENTS = False
GRADIENT_CLIPPING = True
MAX_GRADIENT_NORM = 1.0

# Reprodutibilidade
RANDOM_SEED = 42
SET_SEED = True

# =============================================================================
# CONFIGURAÇÕES DE PERFORMANCE
# =============================================================================

# Otimizações
USE_VECTORIZED_OPERATIONS = True
BATCH_PROCESSING = True

# Limites de memória
MAX_MEMORY_USAGE_MB = 500
GARBAGE_COLLECTION_FREQUENCY = 1000  # Passos entre limpezas

# =============================================================================
# CONFIGURAÇÕES ESPECÍFICAS DE ALGORITMOS
# =============================================================================

# DQN Variants
USE_DOUBLE_DQN = False
USE_DUELING_DQN = False
USE_PRIORITIZED_REPLAY = False

# Policy Gradient (para implementações futuras)
POLICY_GRADIENT_BASELINE = True
ENTROPY_COEFFICIENT = 0.01

# =============================================================================
# CONFIGURAÇÕES DE TESTE
# =============================================================================

# Ambientes de teste
TEST_ENVIRONMENTS = ["gridworld", "cartpole"]
TEST_EPISODES = 100

# Métricas de sucesso
SUCCESS_THRESHOLD = 0.8  # Taxa de sucesso mínima
CONVERGENCE_WINDOW = 100  # Janela para calcular médias

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def get_network_architecture(input_size, output_size):
    """Retorna a arquitetura completa da rede neural."""
    layers = [input_size] + HIDDEN_LAYERS + [output_size]
    return layers

def get_config_summary():
    """Retorna um resumo das configurações principais."""
    summary = {
        "network_layers": HIDDEN_LAYERS,
        "learning_rate": LEARNING_RATE,
        "gamma": GAMMA,
        "epsilon_decay": EPSILON_DECAY,
        "buffer_size": BUFFER_SIZE,
        "batch_size": BATCH_SIZE,
        "max_episodes": MAX_EPISODES
    }
    return summary

def validate_config():
    """Valida se as configurações são consistentes."""
    errors = []
    
    if LEARNING_RATE <= 0:
        errors.append("LEARNING_RATE deve ser positivo")
    
    if GAMMA < 0 or GAMMA > 1:
        errors.append("GAMMA deve estar entre 0 e 1")
    
    if EPSILON_START < EPSILON_END:
        errors.append("EPSILON_START deve ser maior que EPSILON_END")
    
    if BATCH_SIZE > BUFFER_SIZE:
        errors.append("BATCH_SIZE não pode ser maior que BUFFER_SIZE")
    
    if len(HIDDEN_LAYERS) == 0:
        errors.append("Deve haver pelo menos uma camada oculta")
    
    return errors

# Validar configurações na importação
if __name__ == "__main__":
    errors = validate_config()
    if errors:
        print("Erros de configuração encontrados:")
        for error in errors:
            print(f"- {error}")
    else:
        print("Configurações validadas com sucesso!")
        print("\nResumo das configurações:")
        for key, value in get_config_summary().items():
            print(f"- {key}: {value}")
