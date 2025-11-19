# Título

*Projeto 3*

> **Disciplina:** DEC107 — Processamento Paralelo
>
> **Curso:** Bacharelado em Ciência da Computação
>
> **Autores:** João Manoel Fidelis Santos e Maria Eduarda Guedes Alves
>
> **Data:** 28/11/2025

## Arquivos

Referente às funções dgemm e auxiliares:
* ```funcoes.h```: Contém as declarações
* ```funcoes.c```: Contém os códigos das funções
* ```main.c```: Função principal que as chama
* ```Makefile.mak```: Arquivo de compilação que constrói o executável híbrido main
* ```generate_plots.py```: Script Python para analisar os logs (.txt) e gerar os gráficos de desempenho (.png)

## Objetivos do Projeto



## Instalação

O projeto foi desenvolvido para um ambiente Linux (Ubuntu, via WSL) e requer as seguintes bibliotecas:

### 1. Compiladores e MPI:

O projeto usa a biblioteca de MPI chamada Open MPI

~~~bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y build-essential
sudo apt install -y openmpi-bin libopenmpi-dev
~~~
  
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
    
## Compilar, Executar e Analisar

O fluxo de trabalho completo é dividido em três etapas:

### 1. Compilar (Gerar o executável main)

**Use o comando make:**

* Execução padrão com 4 threads

  ~~~bash
  make run
  ~~~

* Mudando número de Threads e tamanho da matriz

  ~~~bash
  make run NP=6 N=4096
  ~~~

**Teste sem make:**

~~~bash
mpicc -O3 -fopenmp -march=native -Wall -o main main.c funcoes.c -lopenblas -lm
~~~

*Isso irá compilar main.c e funcoes.c e criar o executável main.*

### 2. Executar (Gerar o Log de Teste)

* Usando comando make

  ~~~bash
  make run NP=6
  ~~~

* Usando mpirun

  ~~~bash
  mpirun -np 6 ./main
  ~~~

*Usamos 6 processos MPI para corresponder ao número de núcleos físicos do hardware (AMD Ryzen 5 5600G).*

### 3. Analisar (Gerar os Gráficos)

Após a execução, use o script Python para analisar os logs e gerar os gráficos.

~~~bash
python3 generate_plots.py
~~~

O script irá processar automaticamente os arquivos teste6_par_openMP.txt (se existir) e teste7_par_mpi.txt, salvando todos os gráficos .png nas suas respectivas pastas (plots_teste6_par_openMP/ e plots_teste7_par_mpi/).
