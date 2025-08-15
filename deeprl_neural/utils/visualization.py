#!/usr/bin/env python3
"""
Visualization Utils - Utilitários para visualização de dados de treinamento
Implementação em Python puro para gráficos e visualizações simples
"""

import math
from typing import List, Dict, Any, Tuple, Optional


class SimpleChart:
    """
    Classe para criar gráficos simples em texto
    """
    
    def __init__(self, width: int = 60, height: int = 20):
        """
        Inicializa o gerador de gráficos
        
        Args:
            width: Largura do gráfico em caracteres
            height: Altura do gráfico em linhas
        """
        self.width = width
        self.height = height
    
    def plot_line(self, data: List[float], title: str = "Line Chart",
                  x_label: str = "X", y_label: str = "Y") -> str:
        """
        Cria um gráfico de linha simples
        
        Args:
            data: Dados para plotar
            title: Título do gráfico
            x_label: Rótulo do eixo X
            y_label: Rótulo do eixo Y
            
        Returns:
            String com o gráfico em ASCII
        """
        if not data:
            return "No data to plot"
        
        # Normalizar dados
        min_val = min(data)
        max_val = max(data)
        range_val = max_val - min_val if max_val != min_val else 1
        
        # Criar grid
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        
        # Plotar dados
        for i, value in enumerate(data):
            if i >= self.width:
                break
            
            # Normalizar posição Y
            y_normalized = (value - min_val) / range_val
            y_pos = int((self.height - 1) * (1 - y_normalized))
            y_pos = max(0, min(self.height - 1, y_pos))
            
            grid[y_pos][i] = '*'
        
        # Converter para string
        output = [title.center(self.width)]
        output.append('─' * self.width)
        
        for row in grid:
            output.append(''.join(row))
        
        # Adicionar eixos
        output.append('─' * self.width)
        output.append(f"{y_label}: {min_val:.3f} to {max_val:.3f}")
        output.append(f"{x_label}: 0 to {len(data)}")
        
        return '\n'.join(output)
    
    def plot_histogram(self, data: List[float], bins: int = 10,
                      title: str = "Histogram") -> str:
        """
        Cria um histograma simples
        
        Args:
            data: Dados para o histograma
            bins: Número de bins
            title: Título do gráfico
            
        Returns:
            String com o histograma
        """
        if not data:
            return "No data to plot"
        
        # Calcular bins
        min_val = min(data)
        max_val = max(data)
        range_val = max_val - min_val if max_val != min_val else 1
        bin_width = range_val / bins
        
        # Contar frequências
        bin_counts = [0] * bins
        for value in data:
            bin_index = int((value - min_val) / bin_width)
            bin_index = min(bin_index, bins - 1)
            bin_counts[bin_index] += 1
        
        # Normalizar para altura do gráfico
        max_count = max(bin_counts) if bin_counts else 1
        
        output = [title.center(self.width)]
        output.append('─' * self.width)
        
        # Plotar barras
        for i in range(self.height):
            row = ""
            for j, count in enumerate(bin_counts):
                bar_height = int((count / max_count) * self.height)
                if self.height - i <= bar_height:
                    row += "█"
                else:
                    row += " "
                
                if j < len(bin_counts) - 1:
                    row += " "
            
            output.append(row)
        
        # Adicionar labels
        output.append('─' * self.width)
        output.append(f"Range: {min_val:.3f} to {max_val:.3f}")
        output.append(f"Total samples: {len(data)}")
        
        return '\n'.join(output)


class TrainingVisualizer:
    """
    Visualizador específico para dados de treinamento
    """
    
    def __init__(self, chart_width: int = 80, chart_height: int = 15):
        """
        Inicializa o visualizador
        
        Args:
            chart_width: Largura dos gráficos
            chart_height: Altura dos gráficos
        """
        self.chart = SimpleChart(chart_width, chart_height)
    
    def plot_training_progress(self, rewards: List[float], 
                              losses: List[float] = None,
                              success_rates: List[float] = None) -> str:
        """
        Plota progresso do treinamento
        
        Args:
            rewards: Lista de recompensas por episódio
            losses: Lista de losses (opcional)
            success_rates: Lista de taxas de sucesso (opcional)
            
        Returns:
            String com visualizações
        """
        output = []
        
        # Gráfico de recompensas
        if rewards:
            output.append(self.chart.plot_line(
                rewards, 
                "Training Rewards", 
                "Episodes", 
                "Reward"
            ))
            output.append("\n")
        
        # Gráfico de loss
        if losses:
            output.append(self.chart.plot_line(
                losses,
                "Training Loss",
                "Episodes", 
                "Loss"
            ))
            output.append("\n")
        
        # Gráfico de taxa de sucesso
        if success_rates:
            output.append(self.chart.plot_line(
                success_rates,
                "Success Rate",
                "Episodes",
                "Success %"
            ))
            output.append("\n")
        
        return '\n'.join(output)
    
    def plot_q_values_heatmap(self, q_table: List[List[float]], 
                             state_labels: List[str] = None,
                             action_labels: List[str] = None) -> str:
        """
        Plota Q-values como mapa de calor simples
        
        Args:
            q_table: Tabela Q (estados x ações)
            state_labels: Rótulos dos estados
            action_labels: Rótulos das ações
            
        Returns:
            String com mapa de calor
        """
        if not q_table or not q_table[0]:
            return "No Q-table data"
        
        # Normalizar valores para símbolos
        all_values = [val for row in q_table for val in row]
        min_val = min(all_values)
        max_val = max(all_values)
        range_val = max_val - min_val if max_val != min_val else 1
        
        # Símbolos para diferentes intensidades
        symbols = [' ', '░', '▒', '▓', '█']
        
        output = ["Q-Values Heatmap"]
        output.append('─' * 50)
        
        # Headers das ações
        if action_labels:
            header = "State\\Action |"
            for label in action_labels:
                header += f" {label:^3} |"
            output.append(header)
            output.append('─' * len(header))
        
        # Linhas da tabela
        for i, row in enumerate(q_table):
            line = ""
            
            # Label do estado
            if state_labels and i < len(state_labels):
                line += f"{state_labels[i]:^12} |"
            else:
                line += f"State {i:^6} |"
            
            # Valores
            for val in row:
                normalized = (val - min_val) / range_val
                symbol_index = int(normalized * (len(symbols) - 1))
                symbol_index = max(0, min(len(symbols) - 1, symbol_index))
                line += f" {symbols[symbol_index]:^3} |"
            
            output.append(line)
        
        # Legenda
        output.append('─' * 50)
        output.append(f"Legend: {symbols[0]}={min_val:.2f} ... {symbols[-1]}={max_val:.2f}")
        
        return '\n'.join(output)
    
    def create_summary_report(self, training_data: Dict[str, Any]) -> str:
        """
        Cria relatório resumido do treinamento
        
        Args:
            training_data: Dados do treinamento
            
        Returns:
            String com relatório
        """
        output = []
        output.append("=" * 60)
        output.append("TRAINING SUMMARY REPORT".center(60))
        output.append("=" * 60)
        
        # Informações gerais
        if 'total_episodes' in training_data:
            output.append(f"Total Episodes: {training_data['total_episodes']}")
        
        if 'total_steps' in training_data:
            output.append(f"Total Steps: {training_data['total_steps']}")
        
        if 'training_time' in training_data:
            output.append(f"Training Time: {training_data['training_time']:.2f}s")
        
        output.append("")
        
        # Métricas de performance
        if 'final_metrics' in training_data:
            metrics = training_data['final_metrics']
            output.append("PERFORMANCE METRICS:")
            output.append("-" * 20)
            
            for key, value in metrics.items():
                if isinstance(value, float):
                    output.append(f"{key.replace('_', ' ').title()}: {value:.4f}")
                else:
                    output.append(f"{key.replace('_', ' ').title()}: {value}")
        
        output.append("")
        
        # Configuração do agente
        if 'agent_config' in training_data:
            config = training_data['agent_config']
            output.append("AGENT CONFIGURATION:")
            output.append("-" * 20)
            
            for key, value in config.items():
                output.append(f"{key.replace('_', ' ').title()}: {value}")
        
        output.append("")
        output.append("=" * 60)
        
        return '\n'.join(output)


def print_progress_bar(current: int, total: int, width: int = 50,
                      prefix: str = "", suffix: str = "") -> str:
    """
    Cria uma barra de progresso simples
    
    Args:
        current: Progresso atual
        total: Total
        width: Largura da barra
        prefix: Texto antes da barra
        suffix: Texto depois da barra
        
    Returns:
        String com barra de progresso
    """
    if total == 0:
        percent = 100.0
    else:
        percent = 100.0 * current / total
    
    filled_length = int(width * current // total) if total > 0 else width
    bar = '█' * filled_length + '░' * (width - filled_length)
    
    return f"{prefix} |{bar}| {percent:.1f}% {suffix}"


def format_training_stats(stats: Dict[str, float]) -> str:
    """
    Formata estatísticas de treinamento para exibição
    
    Args:
        stats: Dicionário com estatísticas
        
    Returns:
        String formatada
    """
    output = []
    
    for key, value in stats.items():
        formatted_key = key.replace('_', ' ').title()
        
        if isinstance(value, float):
            if 'rate' in key.lower() or 'percentage' in key.lower():
                output.append(f"{formatted_key}: {value:.1%}")
            else:
                output.append(f"{formatted_key}: {value:.4f}")
        else:
            output.append(f"{formatted_key}: {value}")
    
    return ' | '.join(output)


def create_comparison_table(agents_data: List[Dict[str, Any]],
                           metrics: List[str] = None) -> str:
    """
    Cria tabela comparativa entre agentes
    
    Args:
        agents_data: Lista de dados dos agentes
        metrics: Métricas a comparar
        
    Returns:
        String com tabela comparativa
    """
    if not agents_data:
        return "No agents data provided"
    
    if metrics is None:
        metrics = ['avg_reward', 'success_rate', 'avg_episode_length']
    
    # Calcular largura das colunas
    col_width = 12
    name_width = 15
    
    # Cabeçalho
    output = []
    header = f"{'Agent':<{name_width}}"
    for metric in metrics:
        header += f" | {metric.replace('_', ' ').title():^{col_width}}"
    
    output.append(header)
    output.append('─' * len(header))
    
    # Dados dos agentes
    for agent_data in agents_data:
        agent_name = agent_data.get('name', 'Unknown')
        line = f"{agent_name:<{name_width}}"
        
        for metric in metrics:
            value = agent_data.get(metric, 'N/A')
            if isinstance(value, float):
                if 'rate' in metric.lower():
                    line += f" | {value:^{col_width}.1%}"
                else:
                    line += f" | {value:^{col_width}.4f}"
            else:
                line += f" | {str(value):^{col_width}}"
        
        output.append(line)
    
    return '\n'.join(output)


def save_visualization_report(report_data: Dict[str, Any], 
                             filepath: str = "training_report.txt"):
    """
    Salva relatório de visualização em arquivo
    
    Args:
        report_data: Dados do relatório
        filepath: Caminho do arquivo
    """
    visualizer = TrainingVisualizer()
    
    with open(filepath, 'w', encoding='utf-8') as f:
        # Título
        f.write("DEEP LEARNING + REINFORCEMENT LEARNING TRAINING REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        # Resumo
        if 'summary' in report_data:
            f.write(visualizer.create_summary_report(report_data['summary']))
            f.write("\n\n")
        
        # Gráficos de progresso
        if 'rewards' in report_data:
            f.write("TRAINING PROGRESS\n")
            f.write("-" * 20 + "\n")
            f.write(visualizer.plot_training_progress(
                report_data.get('rewards', []),
                report_data.get('losses', []),
                report_data.get('success_rates', [])
            ))
            f.write("\n\n")
        
        # Q-table heatmap
        if 'q_table' in report_data:
            f.write("Q-VALUES VISUALIZATION\n")
            f.write("-" * 20 + "\n")
            f.write(visualizer.plot_q_values_heatmap(
                report_data['q_table'],
                report_data.get('state_labels'),
                report_data.get('action_labels')
            ))
            f.write("\n\n")
        
        # Comparação de agentes
        if 'agents_comparison' in report_data:
            f.write("AGENTS COMPARISON\n")
            f.write("-" * 20 + "\n")
            f.write(create_comparison_table(report_data['agents_comparison']))
            f.write("\n\n")
        
        f.write("End of Report\n")
        f.write("=" * 60 + "\n")


# Funções utilitárias para análise de dados
def calculate_trend(data: List[float], window: int = 10) -> str:
    """
    Calcula tendência dos dados
    
    Args:
        data: Lista de valores
        window: Tamanho da janela para análise
        
    Returns:
        Descrição da tendência
    """
    if len(data) < window * 2:
        return "Insufficient data"
    
    first_half = data[:len(data)//2]
    second_half = data[len(data)//2:]
    
    first_avg = sum(first_half) / len(first_half)
    second_avg = sum(second_half) / len(second_half)
    
    improvement = second_avg - first_avg
    percent_improvement = (improvement / abs(first_avg)) * 100 if first_avg != 0 else 0
    
    if improvement > 0:
        return f"Improving (+{percent_improvement:.1f}%)"
    elif improvement < 0:
        return f"Declining ({percent_improvement:.1f}%)"
    else:
        return "Stable"


def detect_convergence(data: List[float], threshold: float = 0.01,
                      window: int = 50) -> bool:
    """
    Detecta se os dados convergiram
    
    Args:
        data: Lista de valores
        threshold: Threshold para variação
        window: Tamanho da janela
        
    Returns:
        True se convergiu
    """
    if len(data) < window:
        return False
    
    recent_data = data[-window:]
    mean_val = sum(recent_data) / len(recent_data)
    
    # Calcular variação
    variations = [abs(val - mean_val) / abs(mean_val) if mean_val != 0 else 0 
                 for val in recent_data]
    avg_variation = sum(variations) / len(variations)
    
    return avg_variation < threshold
