#ifndef FUNCOES_H
#define FUNCOES_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

#include <omp.h>
#include <cblas.h>
#include <immintrin.h>

#include <mpi.h>

typedef void (*func_matriz)(double* A, double* B, double* C, int tam);

void imprimir_informacoes_iniciais(FILE *file, char nome_projeto[50]);
void imprimir_hardware(FILE *file);
void imprimir_matriz(double *M, int n, FILE *file);
double* aloca_matriz(int n);
void inicializa_matrizes(double *A, double *B, double *C, int n);
void zera_matriz(double *C, int n);

void dgemm_sequencial(double *A, double *B, double *C, int n);
void dgemm_paralelo_openMP(double *A, double *B, double *C, int n);
void dgemm_blas_wrapper(double* A, double* B, double* C, int tam);

void dgemm_paralelo_mpi(
    FILE *file,
    int NUM_REPETICOES,
    double *A_global,
    double *B_global,
    double *C_global_out,
    int tam_matriz,
    int rank,
    int size,
    double tempo_seq_medio
);

double teste(FILE *file, func_matriz funcao, int NUM_REPETICOES, 
             double *A, double *B, double *C, int tam_matriz, char *num_threads, 
             char *speedup, char *eficiencia, double tempo_seq_medio);

void valida_resultado(FILE *file, double *C_correto, double *C_calculado, int n, char *versao);

#endif