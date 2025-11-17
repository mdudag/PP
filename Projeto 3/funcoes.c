#include "funcoes.h"

void dgemm_sequencial(double *A, double *B, double *C, int n) {
  long n_long = n;
  long block = 256;
  long ii, jj, kk;
  long i, j, k;

  for (ii = 0; ii < n_long; ii += block) {
    for (kk = 0; kk < n_long; kk += block) {
      for (jj = 0; jj < n_long; jj += block) {
		for (i = ii; i < ii + block && i < n_long; i++) {
          for (k = kk; k < kk + block && k < n_long; k++) {
            double a_ik = A[n_long * i + k];
            __m256d mA = _mm256_set1_pd(a_ik);
            for (j = jj; j < jj + block && j < n_long; j+=4) {
              __m256d mB = _mm256_loadu_pd(&B[n_long * k + j]);
              __m256d mC = _mm256_loadu_pd(&C[n_long * i + j]);

              mC = _mm256_fmadd_pd(mA, mB, mC);
              _mm256_storeu_pd(&C[n_long * i + j], mC);
            }
          }
        }
      }
    }
  }
}

void dgemm_paralelo_openMP(double *A, double *B, double *C, int n) {
  long n_long = n;
  long block = 256;
  long ii, jj, kk;
  long i, j, k;

  #pragma omp parallel for
  for (ii = 0; ii < n_long; ii += block) {
    for (kk = 0; kk < n_long; kk += block) {
      for (jj = 0; jj < n_long; jj += block) {
		for (i = ii; i < ii + block && i < n_long; i++) {
          for (k = kk; k < kk + block && k < n_long; k++) {
            double a_ik = A[n_long * i + k];
            __m256d mA = _mm256_set1_pd(a_ik);

            for (j = jj; j < jj + block && j < n_long; j+=4) {
              __m256d mB = _mm256_loadu_pd(&B[n_long * k + j]);
              __m256d mC = _mm256_loadu_pd(&C[n_long * i + j]);

              mC = _mm256_fmadd_pd(mA, mB, mC);
              _mm256_storeu_pd(&C[n_long * i + j], mC);
            }
          }
        }
      }
    }
  }
}

double* aloca_matriz(int n) {
  double *M = (double*) malloc((size_t)n * (size_t)n * sizeof(double));
  if (!M) { printf("\nERRO: Falha na alocacao de memoria.\n"); exit(-1); }
  return M;
}

void inicializa_matrizes(double *A, double *B, double *C, int n) {
  long total = (long)n * n;

  #pragma omp parallel 
  {
    unsigned int seed = omp_get_thread_num() + (unsigned int)time(NULL);
  
	#pragma omp for
    for (long i = 0; i < total; i++) {
      A[i] = (double)(rand_r(&seed) % 3 - 1);
      B[i] = (double)(rand_r(&seed) % 9 - 4);
    }
  }
}

void zera_matriz(double *C, int n) {
    long total = (long)n * n;
    #pragma omp parallel for
    for (long i = 0; i < total; i++) {
        C[i] = 0.0;
    }
}

void imprimir_matriz(double *M, int n, FILE *file) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      fprintf(file, "%8.2f ", M[i*n + j]);
    }
    fprintf(file, "\n");
  }
  fprintf(file, "\n");
}

void dgemm_blas_wrapper(double* A, double* B, double* C, int tam) {
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              tam, tam, tam, 
              1.0, A, tam, B, tam, 0.0, C, tam);
}

static void dgemm_local_blocos(double *A_loc, double *B_glob, double *C_loc, int n_global, int local_rows) {
  long n_long = n_global;
  long block = 256;
  long ii, jj, kk;
  long i, j, k;

  for (ii = 0; ii < local_rows; ii += block) {
    for (kk = 0; kk < n_long; kk += block) {
      for (jj = 0; jj < n_long; jj += block) {
        
        for (i = ii; i < ii + block && i < local_rows; i++) {
          for (k = kk; k < kk + block && k < n_long; k++) {
            double a_ik = A_loc[n_long * i + k];
            __m256d mA = _mm256_set1_pd(a_ik);
            
            for (j = jj; j < jj + block && j < n_long; j+=4) {
              __m256d mB = _mm256_loadu_pd(&B_glob[n_long * k + j]);
              __m256d mC = _mm256_loadu_pd(&C_loc[n_long * i + j]);

              mC = _mm256_fmadd_pd(mA, mB, mC);
              _mm256_storeu_pd(&C_loc[n_long * i + j], mC);
            }
          }
        }
      }
    }
  }
}

void dgemm_paralelo_mpi(FILE *file, int NUM_REPETICOES, 
                        double *A_global, double *B_global, double *C_global_out, 
                        int n, int rank, int size, 
                        double tempo_seq_medio) {

  int m = n, k = n;

  int base = m / size;
  int rem = m % size;
  int local_rows = (rank < rem) ? base + 1 : base;

  double *A_loc = (double *) malloc((size_t)local_rows * k * sizeof(double));
  double *C_loc = (double *) calloc((size_t)local_rows * n, sizeof(double));
  double *B_loc = (double *) malloc((size_t)k * n * sizeof(double));
  
  if (!A_loc || !C_loc || !B_loc) {
    fprintf(stderr, "[Rank %d] ERRO: Falha na alocacao de memoria local MPI.\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int *sendcounts = NULL;
  int *displs = NULL;

  if (rank == 0) {
    sendcounts = (int*) malloc(size * sizeof(int));
    displs = (int*) malloc(size * sizeof(int));
    if (!sendcounts || !displs) {
        fprintf(stderr, "[Rank 0] ERRO: Falha na alocacao de metadados MPI.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int offset = 0;
    for (int r = 0; r < size; r++) {
        int rows_r = (r < rem) ? base + 1 : base;
        sendcounts[r] = rows_r * k;
        displs[r] = offset;
        offset += sendcounts[r];
    }
  }

  double tempo_total_mpi = 0.0;
  
  for (int rep = 0; rep < NUM_REPETICOES; rep++) {
      
    if (rank == 0) {
        memset(C_global_out, 0, (size_t)m * n * sizeof(double));
    }
    memset(C_loc, 0, (size_t)local_rows * n * sizeof(double));
    
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Scatterv(A_global,
                 sendcounts,
                 displs,
                 MPI_DOUBLE,
                 A_loc,
                 local_rows * k,
                 MPI_DOUBLE,
                 0,
                 MPI_COMM_WORLD);

    if (rank == 0) {
        memcpy(B_loc, B_global, (size_t)k * n * sizeof(double));
    }
    MPI_Bcast(B_loc, k * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    dgemm_local_blocos(A_loc, B_loc, C_loc, n, local_rows);

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    tempo_total_mpi += (t1 - t0);

    if (rank == 0) {
        int offset = 0;
        for (int r = 0; r < size; r++) {
            int rows_r = (r < rem) ? base + 1 : base;
            sendcounts[r] = rows_r * n;
            displs[r] = offset;
            offset += sendcounts[r];
        }
    }

    MPI_Gatherv(C_loc,
                local_rows * n,
                MPI_DOUBLE,
                C_global_out,
                sendcounts,
                displs,
                MPI_DOUBLE,
                0,
                MPI_COMM_WORLD);

  } 

  if (rank == 0) {
    double tempo_medio_mpi = tempo_total_mpi / NUM_REPETICOES;
    long n_long = n;
    double flops = 2.0 * n_long * n_long * n_long;
    double gflops_mpi = (flops / tempo_medio_mpi) / 1e9;
    
    double speedup_mpi = tempo_seq_medio / tempo_medio_mpi;
    double eficiencia_mpi = (speedup_mpi / (double)size) * 100.0;
    
    char n_procs_str[20];
    char speedup_buf[20];
    char eficiencia_buf[20];
    sprintf(n_procs_str, "%d (MPI)", size);
    sprintf(speedup_buf, "%.6f", speedup_mpi);
    sprintf(eficiencia_buf, "%.6f", eficiencia_mpi);
    
    fprintf(file, "| %-8d | %-8s | %-12.6f | %-10.3f | %-10s | %-10s |\n",
            n, n_procs_str, tempo_medio_mpi, gflops_mpi, speedup_buf, eficiencia_buf);
            
    free(sendcounts);
    free(displs);
  }

  free(A_loc);
  free(C_loc);
  free(B_loc);
}
double teste(FILE *file, func_matriz funcao, int NUM_REPETICOES, double *A, double *B, double *C, 
           int tam_matriz, char *num_threads, char *speedup, char *eficiencia, double tempo_seq_medio) {

  long n_long = tam_matriz;
  double flops = 2.0 * n_long * n_long * n_long;
  double tempo_total = 0.0;

  for (int rep = 0; rep < NUM_REPETICOES; rep++) {
    // Zera a matriz C antes de cada repetição
    zera_matriz(C, tam_matriz);

    double t0 = omp_get_wtime();
    funcao(A, B, C, tam_matriz);
    double t1 = omp_get_wtime();
    tempo_total += (t1 - t0);
  }

  double tempo_medio = tempo_total / NUM_REPETICOES;
  
  if (tempo_seq_medio > 0.0) {
    double n_threads_d = atof(num_threads); 
    double speedup_d = tempo_seq_medio / tempo_medio;
    double eficiencia_d = (speedup_d / n_threads_d) * 100.0;
    
	snprintf(speedup, 20, "%.6f", speedup_d);
    snprintf(eficiencia, 20, "%.6f", eficiencia_d);
  }

  double gflops = (flops / tempo_medio) / 1e9;

  fprintf(file, "| %-8d | %-8s | %-12.6f | %-10.3f | %-10s | %-10s |\n",
          tam_matriz, num_threads, tempo_medio, gflops, speedup, eficiencia);

  if (tempo_seq_medio == 0.0) {
    return tempo_medio;
  }
  return 0;
}

void valida_resultado(FILE *file, double *C_correto, double *C_calculado, int n, char *versao) {
    
    double max_rel_diff = 0.0;
    long n_long = n;
    long total_elementos = n_long * n_long;
    
    // Epsilon para evitar divisão por zero, conforme especificado no PDF [cite: 3791]
    const double epsilon = 1e-12; // Pode ser DBL_EPSILON de <float.h>

    for (long i = 0; i < total_elementos; i++) {
        double C_seq = C_correto[i];
        double C_par = C_calculado[i];
        double diff_abs = fabs(C_seq - C_par);
        double rel_diff;

        // Implementa a fórmula de diferença relativa do PDF [cite: 3791]
        // Usamos o valor absoluto do sequencial + epsilon para segurança
        if (fabs(C_seq) < epsilon) {
            rel_diff = diff_abs; // Se C_seq é ~0, usamos a diferença absoluta
        } else {
            rel_diff = diff_abs / fabs(C_seq);
        }

        if (rel_diff > max_rel_diff) {
            max_rel_diff = rel_diff;
        }
    }
    
    // Imprime o resultado da validação no arquivo de log
    fprintf(file, "[VALIDACAO] %-8s: Diferenca Relativa Maxima = %.6e\n", versao, max_rel_diff);
    
    // Compara com o limite aceitável do PDF [cite: 3792]
    if (max_rel_diff > 1e-9) {
        fprintf(file, "[VALIDACAO] %-8s: FALHOU (Diferenca > 1e-9)\n", versao);
    } else {
        fprintf(file, "[VALIDACAO] %-8s: SUCESSO\n", versao);
    }
}

void imprimir_informacoes_iniciais(FILE *file, char nome_projeto[50]) {
  fprintf(file, "================================================================\n");
  fprintf(file, "%s\n", nome_projeto);
  fprintf(file, "================================================================\n");
  fprintf(file, "Autores: Joao Manoel Fidelis Santos e Maria Eduarda Guedes Alves\n");
  fprintf(file, "Disciplina: DEC107 — Processamento Paralelo\n\n");
  fprintf(file, "Hardware do computador utilizado:\n");
  imprimir_hardware(file);
}

void imprimir_hardware(FILE *file) {
  fprintf(file, " - Placa-mae: SOYO SY-Classic B450M\n");
  fprintf(file, " - CPU: AMD Ryzen 5 5600G (6 nucleos, 12 threads) @ 4.5 GHz\n");
  fprintf(file, " - Cache L2: 3 MB, Cache L3: 16 MB\n");
  fprintf(file, " - RAM: 40 GB DDR4 (Dual Channel) @ 2393 MHz\n");
  fprintf(file, " - Sistema Operacional: Ubuntu 24.04 LTS\n");
  fprintf(file, " - Compilador: GCC 13.3.0\n");
  fprintf(file, "----------------------------------------------------------------\n");
}
