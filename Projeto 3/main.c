#include "funcoes.h"

int main(int argc, char *argv[]) {
  
  int rank, size;
  
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  FILE *file = NULL;

  char nomeFile[100] = "./testes/teste3_mpi_processo2-4.txt";

  if (rank == 0) {
    file = fopen(nomeFile, "w");
    if (!file) {
      printf("Erro ao abrir o arquivo", nomeFile, "\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
      return 1;
    }

    imprimir_informacoes_iniciais(file, "   Projeto 2 — Multiplicacao de Matrizes (DGEMM) com MPI");
  }

  // int tam_matrizes[] = {512, 1024, 2048, 4096};
  int tam_matrizes[] = {512, 1024, 2048};
  int num_tam = sizeof(tam_matrizes) / sizeof(int);
  
  // int contagens_threads[] = {2, 4, 6};
  int contagens_threads[] = {2, 4};
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

    if (rank == 0) {
        printf("--> Testes para matriz %dx%d...\n", tam_matriz, tam_matriz);
        if (tam_matriz == 4096) printf("    (Isso pode demorar alguns minutos no teste sequencial, aguarde...)\n");
        fflush(stdout); // Força a escrita na tela imediatamente
    }

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
        sprintf(n_threads_str, "%d (OpenMP)", n_threads);
        omp_set_num_threads(n_threads);
        
        teste(file, dgemm_paralelo_openMP, NUM_REPETICOES, A, B, C, 
              tam_matriz, n_threads_str, speedup_buf, eficiencia_buf, tempo_seq_medio);

        valida_resultado(file, C_correto, C, tam_matriz, n_threads_str); 
      }
    } 

    MPI_Bcast(&tempo_seq_medio, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    for (int p = 0; p < num_thread_counts; p++) {
        int np_teste = contagens_threads[p];
        
        if (size < np_teste) continue; // Pula se não tiver processos suficientes

        // Cria subgrupo
        int color = (rank < np_teste) ? 0 : MPI_UNDEFINED;
        MPI_Comm sub_comm;
        MPI_Comm_split(MPI_COMM_WORLD, color, rank, &sub_comm);

        if (sub_comm != MPI_COMM_NULL) {
            // 1. Configura o estado global para o Wrapper saber quem é quem
            set_mpi_state(sub_comm);

            // 2. Prepara strings para o log
            char mpi_str[20];
            char speedup_buf[20] = "";
            char eficiencia_buf[20] = "";
            sprintf(mpi_str, "%d (MPI)", np_teste);

            // 3. CHAMA A FUNÇÃO TESTE PADRÃO!
            // O rank 0 do subgrupo vai escrever no arquivo
            // Todos do subgrupo vão executar o cálculo
            teste(file, dgemm_mpi_wrapper, NUM_REPETICOES, A, B, C, 
                  tam_matriz, mpi_str, speedup_buf, eficiencia_buf, tempo_seq_medio);

            // 4. Validação (Apenas Rank 0 Global faz, pois ele tem C_correto)
            if (rank == 0) {
                valida_resultado(file, C_correto, C, tam_matriz, mpi_str);
            }

            // 5. Limpa estado
            set_mpi_state(MPI_COMM_NULL);
            MPI_Comm_free(&sub_comm);
        }
        
        MPI_Barrier(MPI_COMM_WORLD); // Todos esperam o teste terminar
    }

    if (rank == 0) {
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
