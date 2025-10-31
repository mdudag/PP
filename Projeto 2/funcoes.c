#include "funcoes.h"

void dgemm_sequencial(double *A, double *B, double *C, int n) {
  long n_long = n;
  // máximo recomendado para L2 = ≈360, porém 320 é uma boa escolha para facilitar AVX (múltiplo de 8 ou 16, vetores de 256 bits = 4 doubles por registrador)
  // Memoria necessaria=3×(B^2)×8bytes (double)=24B2
  long block = 256;  // Valor adequado para o cache
  long ii, jj, kk;
  long i, j, k;

  // Acessa os blocos
  for (ii = 0; ii < n_long; ii += block) {
    for (kk = 0; kk < n_long; kk += block) {
      for (jj = 0; jj < n_long; jj += block) {
        // Acessa os elementos das matrizes
        for (i = ii; i < ii + block; i++) {
          for (k = kk; k < kk + block; k++) {
            double a_ik = A[n_long * i + k];
            // Utilizando registradores vetoriais AVX
            for (j = jj; j < jj + block; j+=4) {
              // Coloca o mesmo valor de A em todos os slots do registrador
              __m256d mA = _mm256_set1_pd(a_ik);
              // Carrega 4 doubles de B e C na memoria
              __m256d mB = _mm256_loadu_pd(&B[n_long * k + j]);
              __m256d mC = _mm256_loadu_pd(&C[n_long * i + j]);

              // Faz o calculo de multiplicacao e soma
              mC = _mm256_fmadd_pd(mA, mB, mC);
              // Armazena o resultado em C
              _mm256_storeu_pd(&C[n_long * i + j], mC);
            }
          }
        }
      }
    }
  }
}

void imprimir_matriz(double *M, int n, FILE *file) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fprintf(file, "%8.2f ", M[i*n + j]);  // %8.2f para alinhar e mostrar 2 casas decimais
        }
        fprintf(file, "\n");
    }
    fprintf(file, "\n");
}

void dgemm_paralelo(double *A, double *B, double *C, int n) {
  long n_long = n;

  #pragma omp parallel for
  for (long i = 0; i < n_long; i++) {
    for (long k = 0; k < n_long; k++) {
      double a_ik = A[n_long * i + k];

      for (long j = 0; j < n_long; j++) {
        C[n_long * i + j] += a_ik * B[n_long * k + j];
      }
    }
  }
}

// --- Funções auxiliares ---
double* aloca_matriz(int n) {
  double *M = (double*) malloc((size_t)n * (size_t)n * sizeof(double));
  if (!M) { printf("\nERRO: Falha na alocação de memória.\n"); exit(-1); }
  return M;
}

void inicializa_matrizes(double *A, double *B, double *C, int n) {
  long total = (long)n * n;

  #pragma omp parallel 
  {
    // Cria uma semente aleatória para cada thread
    unsigned int seed = omp_get_thread_num() + (unsigned int)time(NULL);
  
    #pragma omp for
    for (long i = 0; i < total; i++) {
      A[i] = (double)(rand_r(&seed) % 3 - 1);
      B[i] = (double)(rand_r(&seed) % 9 - 4);
      C[i] = 0.0;
    }
  }
}

void dgemm_blas_wrapper(double* A, double* B, double* C, int tam) {
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              tam, tam, tam, 
              1.0, A, tam, B, tam, 0.0, C, tam);
}

void teste(FILE *file, func_matriz funcao, int NUM_REPETICOES, double *A, double *B, double *C, 
           int tam_matriz, char *num_threads, char *speedup, char *eficiencia) {

  long n_long = tam_matriz;
  double flops = 2.0 * n_long * n_long * n_long;
  double tempo_total = 0.0;

  for (int rep = 0; rep < NUM_REPETICOES; rep++) {
    inicializa_matrizes(A, B, C, tam_matriz);

    double t0 = omp_get_wtime();
    funcao(A, B, C, tam_matriz);  // Chama a função passada
    double t1 = omp_get_wtime();

    tempo_total += (t1 - t0);
  }

  double tempo_medio = tempo_total / NUM_REPETICOES;
  double gflops = (flops / tempo_medio) / 1e9;

  fprintf(file, "| %-8d | %-8s | %-12.6f | %-10.3f | %-10s | %-10s |\n",
          tam_matriz, num_threads, tempo_medio, gflops, speedup, eficiencia);
}

void imprimir_informacoes_iniciais(FILE *file, char nome_projeto[50]) {
  fprintf(file, "================================================================\n");
  fprintf(file, "%s\n", nome_projeto);
  fprintf(file, "================================================================\n");
  fprintf(file, "Autores: João Manoel Fidelis Santos e Maria Eduarda Guedes Alves\n");
  fprintf(file, "Disciplina: DEC107 — Processamento Paralelo\n\n");
  fprintf(file, "Hardware do computador utilizado:\n");
  imprimir_hardware(file);
}

void imprimir_hardware(FILE *file) {
  fprintf(file, " - Placa-mãe: SOYO SY-Classic B450M\n");
  fprintf(file, " - CPU: AMD Ryzen 5 5600G (6 núcleos, 12 threads) @ 4.5 GHz\n");
  fprintf(file, " - Cache L2: 3 MB, Cache L3: 16 MB\n");
  fprintf(file, " - RAM: 40 GB DDR4 (Dual Channel) @ 2393 MHz\n");
  fprintf(file, " - Sistema Operacional: Ubuntu 24.04 LTS\n");
  fprintf(file, " - Compilador: GCC 13.3.0\n");
  fprintf(file, "----------------------------------------------------------------\n");
}