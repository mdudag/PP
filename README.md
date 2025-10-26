# Trabalho de Processamento Paralelo

Trabalho realizado como creditação da disciplina de Processamento Paralelo, aplicando conceitos de OpenMP e MPI.

**Equipe:**

* Maria Eduarda Guedes Alves
* João Manoel Fidelis Santos

## Projeto 1

### Instalações

* Conjunto de pacotes essenciais para compilação C/C++:

  ~~~bash
  sudo apt-get install build-essential
  ~~~

* Biblioteca BLAS e OpenBLAS:

  ~~~bash
  sudo apt install libblas-dev libopenblas-dev -y
  ~~~

* Biblioteca OpenMP:

  ~~~bash
  sudo apt update
  ~~~

  ~~~bash
  sudo apt install libomp-dev
  ~~~

  Verificar se o compilador C/C++ reconhece o suporte a OpenMP:

  ~~~bash
  echo | cpp -fopenmp -dM | grep -i open
  ~~~

### Versão Sequencial

* Compilar com a BLAS:

  ~~~bash
  gcc -O3 -fopenmp -o vSeq vSeq.c -lblas -lopenblas
  ~~~

* Executar:

  ~~~bash
  ./vSeq
  ~~~
  
### Versão Paralela com OpenMP

* Compilar com a BLAS:

  ~~~bash
  gcc -O3 -fopenmp -o vPar vPar.c -lblas -lopenblas
  ~~~

* Executar:

  ~~~bash
  ./vPar
  ~~~

## Projeto 2

### Instalação

* Biblioteca de MPI chamada Open MPI:

  ~~~bash
  sudo apt install openmpi-bin libopenmpi-dev
  ~~~

### Compilar e Executar

  ~~~bash
  mpicc -o main main.c funcoes.c
  ~~~

  ~~~bash
  mpirun -np 8 ./main
  ~~~