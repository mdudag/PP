OBS: Ainda não testado: makefile e mudança para variáveis globais compartilhdas entre main e cuda

# Multiplicação de Matrizes (DGEMM) com CUDA

*Projeto 3*

> **Disciplina:** DEC107 — Processamento Paralelo
>
> **Curso:** Bacharelado em Ciência da Computação
>
> **Autores:** João Manoel Fidelis Santos e Maria Eduarda Guedes Alves
>
> **Data:** 28/11/2025

## Arquivos

* ```dgemm_cuda.cu```: Contém o código da versão dgemm do CUDA
* ```funcoes.c```: Contém os códigos das funções dgemm anteriores e auxiliares
* ```funcoes.h```: Contém as declarações das funções
* ```main.c```: Função principal que chama as versões dgemm anteriores 

Referente a execução e análise:
* ```Makefile```: Arquivo de compilação que constrói o executável híbrido main
* ```generate_plots.py```: Script Python para analisar os logs (.txt) e gerar os gráficos de desempenho (.png)

## Objetivos do Projeto

Este projeto tem como objetivo principal dar continuidade aos desenvolvimentos anteriores do algoritmo DGEMM (sequencial, OpenMP e MPI), avançando agora para o paradigma de programação em GPU por meio de CUDA. Nesta etapa, busca-se explorar o paralelismo massivo das GPUs e comparar seu desempenho com os demais modelos de execução paralela. Os objetivos específicos são:

  1. Implementar uma versão da multiplicação de matrizes DGEMM utilizando CUDA.
  2. Aplicar conceitos de hierarquia de memória (global, compartilhada, constante e registradores).
  3. Comparar o desempenho obtido com as versões anteriores: sequencial, OpenMP e MPI.
  4. Avaliar eficiência, speedup e escalabilidade entre os diferentes modelos de paralelismo.
  5. Implementar e analisar testes de corretude numérica, considerando possíveis erros de ponto flutuante.

## Instalação

O projeto foi desenvolvido para um ambiente Linux (Ubuntu, via WSL) e requer o seguintes:

### 1. REquisitos para compilar CUDA:

* Criar conta para usar a GPU da estação de trabalho hapi-diagnostico.
* GPU NVIDIA compatível com CUDA.
* CUDA Toolkit instalado (nvcc disponível no PATH).
  
### 2. Biblioteca de Álgebra Linear (BLAS):
    
O projeto usa OpenBLAS para a comparação de desempenho 

~~~bash
sudo apt install -y libopenblas-dev
~~~

### 3. Bibliotecas Python (para gráficos):

* Atualização da lista de pacotes

  ~~~bash
  sudo apt update
  ~~~
	
* Instala o Python e os pacotes necessários usando o gerenciador do sistema (Recomendado para ambientes WSL/Ubuntu modernos)

  ~~~bash
  sudo apt install -y python3-pip python3-pandas python3-matplotlib python3-numpy
  ~~~
    
## Compilar e Executar

**Use o comando make:**

### Versões dgemm Anteriores

* Execução padrão com 6

  ~~~bash
  make run
  ~~~

* Mudando número de Threads

  ~~~bash
  make run NP=4
  ~~~

*Isso irá compilar main.c e funcoes.c, criar o executável main e executá-lo.*

*Usamos 6 processos MPI para corresponder ao número de núcleos físicos do hardware (AMD Ryzen 5 5600G).*

### Versão dgemm CUDA

  ~~~bash
  make run_cuda
  ~~~

*Isso irá compilar dgemm_cuda.cu e funcoes.c, criar o executável dgemm_cuda e executá-lo.*

## Validar

Para cada tamanho N, o programa imprime:

- Tempo CPU  
- Tempo GPU  
- GFLOPS  
- Speedup  
- Validação da saída GPU comparada com o resultado sequencial  

Esse formato facilita análise de desempenho e corretude.

## Analisar (Gerar os Gráficos)

Após a execução, use o script Python para analisar os logs e gerar os gráficos.

~~~bash
python3 generate_plots.py
~~~

(MUDAR)
O script irá processar automaticamente o arquivo teste8_cuda.txt, salvando todos os gráficos .png na sua respectiva pasta plots/plots_teste8_cuda.txt

## Limpar arquivos compilados

~~~bash
make clean
~~~