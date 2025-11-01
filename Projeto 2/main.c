#include "funcoes.h"

int main(void) {
  FILE *file = fopen("./testes/teste7_par.txt", "w");
  if (!file) {
    printf("Erro ao abrir o arquivo.\n");
    return 1;
  }

  imprimir_informacoes_iniciais(file, "   Projeto 2 — Multiplicação de Matrizes (DGEMM) com MPI");

  int tam_matrizes[] = {512, 1024, 2048, 4096};
  int num_tam = sizeof(tam_matrizes) / sizeof(int);
  int contagens_threads[] = {2, 4, 8};
  int num_thread_counts = sizeof(contagens_threads) / sizeof(int);
  const int NUM_REPETICOES = 3; // Media de 3 execucoes

  fprintf(file, "\n--- INÍCIO DOS EXPERIMENTOS (Média de %d execuções) ---\n", NUM_REPETICOES);
  fprintf(file, "| %-8s | %-8s | %-12s | %-10s | %-10s | %-10s |\n",
         "Tamanho", "Threads", "Tempo (s)", "GFLOPS", "Speedup", "Eficiência");
  fprintf(file, "|----------|----------|--------------|------------|------------|------------|\n");

  for (int idx = 0; idx < num_tam; idx++) {
    char speedup_buf[10] = ""; 
    char eficiencia_buf[10] = "";
    char n_threads_str[10] = "";

    int tam_matriz = tam_matrizes[idx];

    double *A = aloca_matriz(tam_matriz);
    double *B = aloca_matriz(tam_matriz);
    double *C = aloca_matriz(tam_matriz);

    inicializa_matrizes(A, B, C, tam_matriz);

    fprintf(file, "\n[LOG] Executando testes para matriz de tamanho %dx%d...\n", tam_matriz, tam_matriz);
    
    // --- 1. Teste Sequencial ---
    strcpy(speedup_buf, "1 (Seq)");
    strcpy(eficiencia_buf, "1.00x");
    strcpy(n_threads_str, "100.00%");
    double tempo_seq_medio = teste(file, dgemm_sequencial, NUM_REPETICOES, 
                                   A, B, C, tam_matriz, speedup_buf, eficiencia_buf, 
                                   n_threads_str, 0);

    // --- 2. Teste Paralelo OpenMP ---
    speedup_buf[0] = '\0';
    eficiencia_buf[0] = '\0';
    for (int t = 0; t < num_thread_counts; t++) {
      int n_threads = contagens_threads[t];
      sprintf(n_threads_str, "%d", n_threads);
      omp_set_num_threads(n_threads);
      teste(file, dgemm_paralelo_openMP, NUM_REPETICOES, A, B, C, 
            tam_matriz, n_threads_str, speedup_buf, eficiencia_buf, tempo_seq_medio);
    }
    
    // --- 3. Teste Paralelo MPI ---
    // for (int t = 0; t < num_thread_counts; t++) {
      // int n_threads = contagens_threads[t];
      // sprintf(n_threads_str, "%d", n_threads);
      // omp_set_num_threads(n_threads);
      // teste(file, dgemm_paralelo_mpi, NUM_REPETICOES, A, B, C, 
      //       tam_matriz, n_threads_str, speedup_buf, eficiencia_buf, tempo_seq_medio);
    // }
    
    // --- 4. Teste BLAS ---
    strcpy(speedup_buf, "BLAS");
    strcpy(eficiencia_buf, "-");
    strcpy(n_threads_str, "-");
    teste(file, dgemm_blas_wrapper, NUM_REPETICOES, A, B, C,
          tam_matriz, speedup_buf, eficiencia_buf, n_threads_str, 0);

    fprintf(file, "|----------|----------|--------------|------------|------------|------------|\n");
    free(A); free(B); free(C);
  }

  fprintf(file, "\n[LOG] Experimentos finalizados.\n");
  return 0;
}
