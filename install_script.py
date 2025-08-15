#!/usr/bin/env python3
"""
Script para facilitar a instalação da biblioteca deeprl-neural
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(cmd, cwd=None):
    """Execute um comando e retorna o resultado"""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, 
                              capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao executar: {cmd}")
        print(f"   {e.stderr}")
        return None

def build_library():
    """Constrói o wheel da biblioteca"""
    print("🔨 Construindo a biblioteca...")
    
    # Verificar se o build está disponível
    try:
        import build
    except ImportError:
        print("📦 Instalando build...")
        run_command("pip install build")
    
    # Limpar builds anteriores
    dist_dir = Path("dist")
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    
    build_dir = Path("build")
    if build_dir.exists():
        shutil.rmtree(build_dir)
    
    # Construir
    result = run_command("python -m build")
    if result is not None:
        print("✅ Biblioteca construída com sucesso!")
        return True
    return False

def install_locally():
    """Instala a biblioteca localmente em modo de desenvolvimento"""
    print("📥 Instalando biblioteca em modo de desenvolvimento...")
    result = run_command("pip install -e .")
    if result is not None:
        print("✅ Biblioteca instalada em modo de desenvolvimento!")
        return True
    return False

def install_wheel():
    """Instala a biblioteca via wheel"""
    print("📦 Instalando biblioteca via wheel...")
    
    # Encontrar o wheel gerado
    dist_dir = Path("dist")
    wheels = list(dist_dir.glob("*.whl"))
    
    if not wheels:
        print("❌ Nenhum wheel encontrado. Execute build primeiro.")
        return False
    
    wheel_path = wheels[0]  # Usar o primeiro wheel encontrado
    result = run_command(f"pip install {wheel_path}")
    if result is not None:
        print("✅ Biblioteca instalada via wheel!")
        return True
    return False

def test_installation():
    """Testa se a instalação funcionou"""
    print("🧪 Testando instalação...")
    
    test_code = """
import deeprl_neural
from deeprl_neural import GridWorld, QLearningAgent, NeuralNetwork

print(f"OK deeprl-neural v{deeprl_neural.__version__} importado com sucesso!")

# Teste básico de funcionalidade
env = GridWorld(grid_size=3)
agent = QLearningAgent(9, 4)
nn = NeuralNetwork([2, 4, 1])

print("OK Todos os componentes funcionando!")
"""
    
    try:
        result = subprocess.run([sys.executable, "-c", test_code], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Teste de instalação falhou:")
        print(e.stderr)
        return False

def main():
    """Função principal"""
    print("🚀 Script de Instalação - deeprl-neural")
    print("=" * 50)
    
    if len(sys.argv) < 2:
        print("""
Uso: python install_script.py [opção]

Opções:
  build     - Constrói o wheel da biblioteca
  dev       - Instala em modo de desenvolvimento (pip install -e .)
  wheel     - Constrói e instala via wheel
  test      - Testa a instalação atual
  all       - Faz tudo (build + install + test)

Exemplos:
  python install_script.py dev      # Para desenvolvimento
  python install_script.py wheel    # Para instalação via wheel
  python install_script.py all      # Processo completo
""")
        return
    
    option = sys.argv[1].lower()
    
    if option == "build":
        build_library()
    
    elif option == "dev":
        install_locally()
        test_installation()
    
    elif option == "wheel":
        if build_library():
            install_wheel()
            test_installation()
    
    elif option == "test":
        test_installation()
    
    elif option == "all":
        if build_library():
            install_wheel()
            test_installation()
    
    else:
        print(f"❌ Opção '{option}' não reconhecida.")
        print("Use: build, dev, wheel, test, ou all")

if __name__ == "__main__":
    main()
