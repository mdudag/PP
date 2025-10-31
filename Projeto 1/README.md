# Multiplicação de Matrizes (DGEMM) Sequencial e Paralela com OpenMP
*Projeto 2*

> **Disciplina:** DEC107 — Processamento Paralelo
>
> **Curso:** Bacharelado em Ciência da Computação
>
> **Autores:** João Manoel Fidelis Santos e Maria Eduarda Guedes Alves
>
> **Data:** 27/09/2025

## Instalar

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

## Compilar e Executar

* Versão Sequencial

  ~~~bash
  gcc -O3 -fopenmp -o vSeq vSeq.c -lblas -lopenblas
  ~~~

  ~~~bash
  ./vSeq
  ~~~
  
* Versão Paralela com OpenMP

  ~~~bash
  gcc -O3 -fopenmp -o vPar vPar.c -lblas -lopenblas
  ~~~

  ~~~bash
  ./vPar
  ~~~