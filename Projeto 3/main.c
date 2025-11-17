#include "funcoes.h"

int main(int argc, char *argv[]) {
  
  int rank, size;
  
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  FILE *file = NULL;

  if (rank == 0) {
    file = fopen("./testes/teste7_par_mpi.txt", "w");
    if (!file) {
      printf("Erro ao abrir o arquivo ./testes/teste7_par_mpi.txt\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
      return 1;
    }

    imprimir_informacoes_iniciais(file, "   Projeto 2 — Multiplicacao de Matrizes (DGEMM) com MPI");
  }

  int tam_matrizes[] = {512, 1024, 2048, 4096};
  int num_tam = sizeof(tam_matrizes) / sizeof(int);
  
  int contagens_threads[] = {2, 4, 6};
  int num_thread_counts = sizeof(contagens_threads) / sizeof(int);
  
  const int NUM_REPETICOES = 3;

  if (rank == 0) {
    fprintf(file, "\n--- INICIO DOS EXPERIMENTOS (Media de %d execucoes) ---\n", NUM_REPETICOES);
    fprintf(file, "| %-8s | %-8s | %-12s | %-10s | %-10s | %-10s |\n",
           "Tamanho", "Threads", "Tempo (s)", "GFLOPS", "Speedup", "Eficiencia");
    fprintf(file, "|----------|----------|--------------|------------|------------|------------|\n");
  }

  for (int idx = 0; idx < num_tam; idx++) {
    
    char speedup_buf[20] = ""; 
    char eficiencia_buf[20] = "";
    char n_threads_str[20] = "";

    int tam_matriz = tam_matrizes[idx];

	double *A = NULL;
    double *B = NULL;
    double *C = NULL;
    double *C_correto = NULL; 
    
    double tempo_seq_medio = 0.0;

    if (rank == 0) {
      A = aloca_matriz(tam_matriz);
      B = aloca_matriz(tam_matriz);
      C = aloca_matriz(tam_matriz);
      C_correto = aloca_matriz(tam_matriz); 

      inicializa_matrizes(A, B, C, tam_matriz);

      fprintf(file, "\n[LOG] Executando testes para matriz de tamanho %dx%d...\n", tam_matriz, tam_matriz);
      
      strcpy(n_threads_str, "1 (Seq)");
      strcpy(speedup_buf, "1.00x");
      strcpy(eficiencia_buf, "100.00%");
      tempo_seq_medio = teste(file, dgemm_sequencial, NUM_REPETICOES, 
                                     A, B, C, tam_matriz, n_threads_str, speedup_buf, 
                                     eficiencia_buf, 0);
      
      memcpy(C_correto, C, (size_t)tam_matriz * tam_matriz * sizeof(double));


      for (int t = 0; t < num_thread_counts; t++) {
        speedup_buf[0] = '\0';
        eficiencia_buf[0] = '\0';
        
        int n_threads = contagens_threads[t];
        sprintf(n_threads_str, "%d", n_threads);
        omp_set_num_threads(n_threads);
        
        teste(file, dgemm_paralelo_openMP, NUM_REPETICOES, A, B, C, 
              tam_matriz, n_threads_str, speedup_buf, eficiencia_buf, tempo_seq_medio);

        valida_resultado(file, C_correto, C, tam_matriz, n_threads_str); 
      }
    } 

    MPI_Bcast(&tempo_seq_medio, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    dgemm_paralelo_mpi(file, NUM_REPETICOES, A, B, C, 
                       tam_matriz, rank, size, tempo_seq_medio);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
      char mpi_str[20];
      sprintf(mpi_str, "%d (MPI)", size);
      valida_resultado(file, C_correto, C, tam_matriz, mpi_str);

      strcpy(n_threads_str, "BLAS");
      strcpy(speedup_buf, "-");
      strcpy(eficiencia_buf, "-");

      openblas_set_num_threads(1);  // Blas seqencial
      teste(file, dgemm_blas_wrapper, NUM_REPETICOES, A, B, C,
            tam_matriz, n_threads_str, speedup_buf, eficiencia_buf, 0.0);
            
      valida_resultado(file, C_correto, C, tam_matriz, n_threads_str);

      fprintf(file, "|----------|----------|--------------|------------|------------|------------|\n");
      
      free(A); free(B); free(C); free(C_correto); 
    }
  } 

  if (rank == 0) {
    fprintf(file, "\n[LOG] Experimentos finalizados.\n");
    fclose(file);
  }

  MPI_Finalize();
  
  return 0;
}
