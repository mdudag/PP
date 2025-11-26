#ifndef FUNCOES_H
#define FUNCOES_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

#include <omp.h>
#include <mpi.h>
#include <cblas.h>
#include <immintrin.h>

// Variáveis globais compartilhdas entre cuda e main
extern int tam_matrizes[];
extern int num_tam;
extern const int NUM_REPETICOES;

// Tamanho do bloco de cache
#define SIZE_BLOCK 128

// Definicao do tipo de funcao
typedef void (*func_matriz)(double* A, double* B, double* C, int tam);

// Funcoes auxiliares de IO
void imprimir_informacoes_iniciais(FILE *file, char nome_projeto[50]);
void imprimir_hardware(FILE *file);
void imprimir_matriz(double *M, int n, FILE *file);

// Funcoes auxiliares das funcoes dgemm
double* aloca_matriz(int n);
void inicializa_matrizes(double *A, double *B, double *C, int n);
void zera_matriz(double *C, int n);
void dgemm_local_blocos(double *A, double *B, double *C, int n, int l_rows); 

// Funcoes dgemm
void dgemm_sequencial(double *A, double *B, double *C, int n);
void dgemm_paralelo_openMP(double *A, double *B, double *C, int n);
void dgemm_mpi_wrapper(double *A, double *B, double *C, int n);
void dgemm_blas_wrapper(double* A, double* B, double* C, int tam);

// Funcoes de medicao e validacao
double medir_tempo_execucao(func_matriz funcao, int NUM_REPETICOES, double *A, double *B, double *C, int tam_matriz);
void set_mpi_state(MPI_Comm comm);
double valida_resultado(double *C_correto, double *C_calculado, int n);
void registrar_resultado(FILE *file, int tam_matriz, char *versao, 
                         double tempo, double tempo_seq_base, int n_threads, 
                         double erro_max);

#endif
