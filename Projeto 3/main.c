#include "funcoes.h"

int main(int argc, char *argv[]) {
  int rank, size;
  
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Inicia arquivo
  FILE *file = NULL;
  char nomeFile[100] = "./testes/teste8_cuda.txt";

  if (rank == 0) {
    file = fopen(nomeFile, "w");
    if (!file) {
      printf("Erro ao abrir o arquivo %s\n", nomeFile);
      MPI_Abort(MPI_COMM_WORLD, 1);
      return 1;
    }

    imprimir_informacoes_iniciais(file, "   Projeto 3 — Multiplicação de Matrizes (DGEMM) com CUDA");
      fprintf(file, "----------------------------------------------------------------\n");
      fprintf(file, "                          DGEMM ANTERIORES                      \n");
      fprintf(file, "----------------------------------------------------------------\n");
  }
  
  int contagens_threads[] = {2, 4, 6};
  // int contagens_threads[] = {2, 4}; // Teste no pc de Duda
  int num_thread_counts = sizeof(contagens_threads) / sizeof(int);

  if (rank == 0) {
    fprintf(file, "\n--- INICIO DOS EXPERIMENTOS (Media de %d execucoes) ---\n", NUM_REPETICOES);
    fprintf(file, "| %-8s | %-12s | %-12s | %-10s | %-10s | %-10s | %-12s | %-9s |\n",
           "Tamanho", "Versao", "Tempo (s)", "GFLOPS", "Speedup", "Eficiencia", "Diff. Max", "Validacao");
    fprintf(file, "|----------|--------------|--------------|------------|------------|------------|--------------|-----------|\n");
  }

  for (int idx = 0; idx < num_tam; idx++) {
    int tam_matriz = tam_matrizes[idx];

    // Visualizacao no terminal
    if (rank == 0) {
        printf("--> Testes para matriz %dx%d...\n", tam_matriz, tam_matriz);
        if (tam_matriz == 4096) printf("    (Isso pode demorar alguns minutos, aguarde...)\n");
        fflush(stdout); // Força a escrita na tela imediatamente
    }

	  double *A = NULL;
    double *B = NULL;
    double *C = NULL;
    double *C_correto = NULL; 
    
    double tempo_seq = 0.0;

    if (rank == 0) {
      A = aloca_matriz(tam_matriz);
      B = aloca_matriz(tam_matriz);
      C = aloca_matriz(tam_matriz);
      C_correto = aloca_matriz(tam_matriz); 

      inicializa_matrizes(A, B, C, tam_matriz);
      
      // --- 1. Versao Sequencial ---

      tempo_seq = medir_tempo_execucao(dgemm_sequencial, NUM_REPETICOES, A, B, C, tam_matriz);
      memcpy(C_correto, C, (size_t)tam_matriz * tam_matriz * sizeof(double));
      registrar_resultado(file, tam_matriz, "1 (Seq)", tempo_seq, tempo_seq, 1, 0.0);

      // --- 2. Versao com OpenMP ---

      for (int t = 0; t < num_thread_counts; t++) {
        int n_th = contagens_threads[t];
        char nome_versao[20];
        sprintf(nome_versao, "%d (OpenMP)", n_th);
        
        omp_set_num_threads(n_th);
        
        // Executa, valida e registra
        double tempo = medir_tempo_execucao(dgemm_paralelo_openMP, NUM_REPETICOES, A, B, C, tam_matriz);
        double erro = valida_resultado(C_correto, C, tam_matriz);
        registrar_resultado(file, tam_matriz, nome_versao, tempo, tempo_seq, n_th, erro);
      }
    } 

    // --- 3. Versao com MPI ---
    MPI_Bcast(&tempo_seq, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    for (int p = 0; p < num_thread_counts; p++) {
        int np_teste = contagens_threads[p];
        
        // Se não houver processos suficientes lançados, pula
        if (size < np_teste) continue; 

        // Criação de Sub-comunicador para testar 2, 4, 6 processos isoladamente
        int color = (rank < np_teste) ? 0 : MPI_UNDEFINED;
        MPI_Comm sub_comm;
        MPI_Comm_split(MPI_COMM_WORLD, color, rank, &sub_comm);

        if (sub_comm != MPI_COMM_NULL) {
            // Passa o sub-comunicador para o wrapper saber quem é quem neste grupo
            set_mpi_state(sub_comm);
            
            // Todos do grupo calculam
            double tempo = medir_tempo_execucao(dgemm_mpi_wrapper, NUM_REPETICOES, A, B, C, tam_matriz);

            // Validação e Registro
            int sub_rank;
            MPI_Comm_rank(sub_comm, &sub_rank);

            if (rank == 0 && sub_rank == 0) {
                char nome_versao[20];
                sprintf(nome_versao, "%d (MPI)", np_teste);
                
                double erro = valida_resultado(C_correto, C, tam_matriz);
                
                registrar_resultado(file, tam_matriz, nome_versao, tempo, tempo_seq, np_teste, erro);
            }

            // Limpeza do estado MPI
            set_mpi_state(MPI_COMM_NULL);
            MPI_Comm_free(&sub_comm);
        }
        // Sincroniza o mundo todo antes de ir para o próximo teste MPI
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // --- 4. Versao com BLAS ---

    if (rank == 0) {
      openblas_set_num_threads(1); // Blas sequencial
      
      // Executa, valida e registra
      double tempo = medir_tempo_execucao(dgemm_blas_wrapper, NUM_REPETICOES, A, B, C, tam_matriz);
      double erro = valida_resultado(C_correto, C, tam_matriz);
      registrar_resultado(file, tam_matriz, "BLAS", tempo, 0.0, 0, erro);

      fprintf(file, "|----------|--------------|--------------|------------|------------|------------|--------------|-----------|\n");
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
