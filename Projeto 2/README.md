# Multiplicação de Matrizes (DGEMM) com MPI
*Projeto 2*

> **Disciplina:** DEC107 — Processamento Paralelo
>
> **Curso:** Bacharelado em Ciência da Computação
>
> **Autores:** João Manoel Fidelis Santos e Maria Eduarda Guedes Alves
>
> **Data:** 03/11/2025

## Arquivos

Referente as funções dgemm e auxiliares:
* ```funcoes.h```: Contém as declarações
* ```funcoes.c```: Contém os códigos das funções
* ```main.c```: Função principal que as chamam

## Objetivos do Projeto

* Implementar uma versão paralela distribuída da DGEMM utilizando MPI, explorando a comunicação entre processos distintos via troca de mensagens. 

* Avaliar o desempenho obtido em comparação com: 
  * A versão sequencial (referência de desempenho e corretude);
  * A versão paralela com OpenMP (comparação de speedup e eficiência em diferentes arquiteturas e modelos de paralelismo). 

* Incluir testes de corretude e análise de precisão numérica, ausentes no Projeto 1. 