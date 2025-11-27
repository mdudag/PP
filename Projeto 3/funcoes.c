#include "funcoes.h"

// Variáveis globais compartilhdas entre cuda e main
int tam_matrizes[] = {512, 1024, 2048, 4096};
// int tam_matrizes[] = {512, 1024, 2048}; // Teste rapido
int num_tam = sizeof(tam_matrizes)/sizeof(int);
const int NUM_REPETICOES = 3;

// Variaveis globais de controle para mpi wrapper
static MPI_Comm g_mpi_comm = MPI_COMM_NULL;
static int g_mpi_rank = 0;
static int g_mpi_size = 1;

// --- Funcoes DGEMM ---

void dgemm_sequencial(double *A, double *B, double *C, int n) {
  long n_long = n;
  long block = SIZE_BLOCK;
  long ii, jj, kk;
  long i, j, k;

  // Cache block + registradores AVX

  // Iteram sobre os blocos
  for (ii = 0; ii < n_long; ii += block) {
    for (kk = 0; kk < n_long; kk += block) {
      for (jj = 0; jj < n_long; jj += block) {
        // Iteram dentro dos blocos sobre os doubles
		    for (i = ii; i < ii + block && i < n_long; i++) {
          for (k = kk; k < kk + block && k < n_long; k++) {
            // Guarda um valor de A
            double a_ik = A[n_long * i + k];
            // Copia o mesmo valor 4 vezes em um registrador
            __m256d mA = _mm256_set1_pd(a_ik);
            for (j = jj; j < jj + block && j < n_long; j+=4) {
              // Copia 4 doubles de B em um registrador
              __m256d mB = _mm256_loadu_pd(&B[n_long * k + j]);
              // Copia 4 doubles de C em um registrador
              __m256d mC = _mm256_loadu_pd(&C[n_long * i + j]);
              
              // Faz o calculo de um double de C em uma unica instrucao
              mC = _mm256_fmadd_pd(mA, mB, mC);
              // Salva o resultado na memoria
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
  long block = SIZE_BLOCK;
  long ii, jj, kk;
  long i, j, k;

  // Divide de forma igual as linhas de blocos entre as threads criadas e priva as outras variáveis
  #pragma omp parallel for private(kk, jj, i, j, k) schedule(static)
  for (ii = 0; ii < n_long; ii += block) {
    for (kk = 0; kk < n_long; kk += block) {
      for (jj = 0; jj < n_long; jj += block) {
        long i_end = (ii + block > n_long)? n_long: ii + block;
        long k_end = (kk + block > n_long)? n_long: kk + block;
        long j_end = (jj + block > n_long)? n_long: jj + block;

        for (i = ii; i < i_end; i++) {
          for (k = kk; k < k_end; k++) {
            double a_ik = A[n_long * i + k];
            __m256d mA = _mm256_set1_pd(a_ik);
            
            for (j = jj; j < j_end; j+=4) {
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

void dgemm_local_blocos(double *A_loc, double *B_glob, double *C_loc, int n_global, int local_rows) {
  long n_long = n_global;
  long block = SIZE_BLOCK;
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

void dgemm_mpi_wrapper(double *A_global, double *B_global, double *C_global, int n) {
    if (g_mpi_comm == MPI_COMM_NULL) return;

    int rank = g_mpi_rank;
    int size = g_mpi_size;
    
    int m = n, k = n;
    int base = m / size;  // Linhas de cada processo
    int rem = m % size;   // Linhas que sobraram
    // Distrubui as linhas que sobraram aos rem primeiros processos
    int local_rows = (rank < rem) ? base + 1 : base;  

    // Alocacao de memoria local
    double *A_loc = (double *) malloc((size_t)local_rows * k * sizeof(double));
    double *C_loc = (double *) calloc((size_t)local_rows * n, sizeof(double));
    double *B_loc = (double *) malloc((size_t)k * n * sizeof(double));

    if (!A_loc || !C_loc || !B_loc) {
      fprintf(stderr, "Erro : Falha de alocacao de memoria no Rank %d\n", rank);
      MPI_Abort(MPI_COMM_WORLD, -1);
    }

    // Inicializa com endereço para calar o warning do compilador se rank não for 0
    int *sendcounts = NULL;
    int *displs = NULL;
    
    if (rank == 0) {
      // Aloca vetores de controle
        sendcounts = (int*) malloc(size * sizeof(int));
        displs = (int*) malloc(size * sizeof(int));
        
        int offset = 0;
        for (int r = 0; r < size; r++) {
            int r_rows = (r < rem) ? base + 1 : base;
            sendcounts[r] = r_rows * k; // Quantidade de linhas para enviar
            displs[r] = offset;         // Deslocamento
            offset += sendcounts[r];
        }
    }

    // Distribuição de A
    MPI_Scatterv(A_global, sendcounts, displs, MPI_DOUBLE, 
                 A_loc, local_rows * k, MPI_DOUBLE, 0, g_mpi_comm);

    // Broadcast de B
    if (rank == 0) memcpy(B_loc, B_global, (size_t)k * n * sizeof(double));
    MPI_Bcast(B_loc, k * n, MPI_DOUBLE, 0, g_mpi_comm);

    // Calculo local
    dgemm_local_blocos(A_loc, B_loc, C_loc, n, local_rows);

    // Rank 0 recalcula os vetores de controle e recolhe C
    if (rank == 0) {
        int offset = 0;
        for (int r = 0; r < size; r++) {
            int r_rows = (r < rem) ? base + 1 : base;
            sendcounts[r] = r_rows * n; 
            displs[r] = offset;
            offset += sendcounts[r];
        }
    }

    MPI_Gatherv(C_loc, local_rows * n, MPI_DOUBLE, 
                C_global, sendcounts, displs, MPI_DOUBLE, 0, g_mpi_comm);

    // Limpeza
    free(A_loc); free(B_loc); free(C_loc);
    
    // Rank 0 libera vetores 
    if (rank == 0) { free(sendcounts); free(displs); }
}

void dgemm_blas_wrapper(double* A, double* B, double* C, int tam) {
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              tam, tam, tam, 
              1.0, A, tam, B, tam, 0.0, C, tam);
}

// -- Funcoes de Medicao e Validacao

double medir_tempo_execucao(func_matriz funcao, int NUM_REPETICOES, double *A, double *B, double *C, int tam_matriz) {
  int is_mpi = (g_mpi_comm != MPI_COMM_NULL);
  double tempo_total = 0.0;

  for (int rep = 0; rep < NUM_REPETICOES; rep++) {
    // Limpa a matriz C para novo calculo
    if (!is_mpi || g_mpi_rank == 0) zera_matriz(C, tam_matriz);
    
    // Barreira inicial para esperar todos os processos
    if (is_mpi) MPI_Barrier(g_mpi_comm);
    double t0 = omp_get_wtime();
    
    funcao(A, B, C, tam_matriz);  // executa uma versao
    
    // Barreira final para esperar todos os processos
    if (is_mpi) MPI_Barrier(g_mpi_comm);
    double t1 = omp_get_wtime();
    
    tempo_total += (t1 - t0);
  }

  return tempo_total / NUM_REPETICOES;
}

double valida_resultado(double *C_correto, double *C_calculado, int n) {
  double max_rel_diff = 0.0;
  long total_elementos = (long)n * n;
  
  double C_seq, C_par, diff_abs, rel_diff;

  for (long i = 0; i < total_elementos; i++) {
    C_seq = C_correto[i];
    C_par = C_calculado[i];
    diff_abs = fabs(C_seq - C_par);

    if (fabs(C_seq) < EPSILON) rel_diff = diff_abs;
    else rel_diff = diff_abs / fabs(C_seq);

    if (rel_diff > max_rel_diff) max_rel_diff = rel_diff;
  }
  return max_rel_diff;
}

void registrar_resultado(FILE *file, int tam_matriz, char *versao, 
                         double tempo, double tempo_seq_base, int n_threads, 
                         double erro_max) {
    
    // Define status da validacao
    char *status_val = (erro_max < TOLERANCIA) ? "SUCESSO" : "FALHOU";
    
    // Calcula GFLOPS
    double gflops = (2.0 * tam_matriz * tam_matriz * tam_matriz / tempo) / 1e9;

    // Formata e imprime na tabela
    // BLAS com n_threads=0
    if (n_threads == 0) {
        fprintf(file, "| %-8d | %-12s | %-12.6f | %-10.3f | %-10s | %-10s | %-12.2e | %-9s |\n",
              tam_matriz, versao, tempo, gflops, "-", "-", erro_max, status_val);
    } 
    // Sequencial
    else if (tempo_seq_base <= 0.0 || strcmp(versao, "1 (Seq)") == 0) {
        fprintf(file, "| %-8d | %-12s | %-12.6f | %-10.3f | %-10s | %-10s | %-12.2e | %-9s |\n",
              tam_matriz, versao, tempo, gflops, "1.00x", "100.00%", erro_max, status_val);
    } 
    // Paralelo (OpenMP ou MPI)
    else {
        double speedup = tempo_seq_base / tempo;
        double eficiencia = (speedup / n_threads) * 100.0;

        fprintf(file, "| %-8d | %-12s | %-12.6f | %-10.3f | %-10.2fx | %-9.2f%% | %-12.2e | %-9s |\n",
              tam_matriz, versao, tempo, gflops, speedup, eficiencia, erro_max, status_val);
    }
}

// --- Funcoes Auxiliares ---
  
// Funcao auxiliar para configurar o ambiente antes de chamar o teste
void set_mpi_state(MPI_Comm comm) {
  if (comm != MPI_COMM_NULL) {
    g_mpi_comm = comm;
    MPI_Comm_rank(comm, &g_mpi_rank);
    MPI_Comm_size(comm, &g_mpi_size);
  } else {
    g_mpi_comm = MPI_COMM_NULL;
    g_mpi_rank = 0;
    g_mpi_size = 1;
  }
}

double* aloca_matriz(int n) {
  double *M = (double*) malloc((size_t)n * (size_t)n * sizeof(double));
  if (!M) { printf("\nERRO: Falha na alocacao de memoria.\n"); exit(-1); }
  return M;
}

void inicializa_matrizes(double *A, double *B, int n, int id) {
  long total = (long)n * n;
  // Gera matrizes iguais, mas diferente para cada tamanho
  unsigned int base_seed = 12345 + n*10 + id;

  #pragma omp parallel 
  {
    unsigned int seed = base_seed + omp_get_thread_num(); // Cada thread tem seu seed
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
