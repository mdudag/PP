#include "funcoes.h"

int main(void) {
  FILE *file = fopen("teste-seq1-cache-blocking.txt", "w");
  if (!file) {
    printf("Erro ao abrir o arquivo.\n");
    return 1;
  }

  imprimir_informacoes_iniciais(file, "   Projeto 2 — Multiplicação de Matrizes (DGEMM) com MPI");

  int tam_matrizes[] = {512, 1024, 2048, 4096};
  int num_tam = sizeof(tam_matrizes) / sizeof(int);
//   int contagens_threads[] = {2, 4, 8};
//   int num_thread_counts = sizeof(contagens_threads) / sizeof(int);
  const int NUM_REPETICOES = 3; // Média de 3 execuções

  fprintf(file, "\n--- INÍCIO DOS EXPERIMENTOS (Média de %d execuções) ---\n", NUM_REPETICOES);
  fprintf(file, "| %-8s | %-8s | %-12s | %-10s | %-10s | %-10s |\n",
         "Tamanho", "Threads", "Tempo (s)", "GFLOPS", "Speedup", "Eficiência");
  fprintf(file, "|----------|----------|--------------|------------|------------|------------|\n");

  for (int idx = 0; idx < num_tam; idx++) {
    int tam_matriz = tam_matrizes[idx];

    double *A = aloca_matriz(tam_matriz);
    double *B = aloca_matriz(tam_matriz);
    double *C = aloca_matriz(tam_matriz);

    inicializa_matrizes(A, B, C, tam_matriz);

    fprintf(file, "\n[LOG] Executando testes para matriz de tamanho %dx%d...\n", tam_matriz, tam_matriz);
    
    // --- 1. Teste Sequencial ---
    teste(file, dgemm_sequencial, NUM_REPETICOES, A, B, C, tam_matriz, "1 (Seq)", "1.00x", "100.00%");

    // --- 2. Teste Paralelo OpenMP ---
    // for (int t = 0; t < num_thread_counts; t++) {
    //   int n_threads = contagens_threads[t];
    //   omp_set_num_threads(n_threads);
    //   teste(file, dgemm_paralelo, NUM_REPETICOES, A, B, C, tam_matriz, "4", "-", "-");
    // }
    
    // --- 3. Teste Paralelo MPI ---
    // for (int t = 0; t < num_thread_counts; t++) {
    //   int n_threads = contagens_threads[t];
    //   omp_set_num_threads(n_threads);
    //   teste(file, dgemm_mpi, NUM_REPETICOES, A, B, C, tam_matriz, "BLAS", "-", "-");
    // }
    
    // --- 4. Teste BLAS ---
    // teste(file, dgemm_blas_wrapper, NUM_REPETICOES, A, B, C, tam_matriz, "BLAS", "-", "-");

    fprintf(file, "|----------|----------|--------------|------------|------------|------------|\n");
    free(A); free(B); free(C);
  }

  fprintf(file, "\n[LOG] Experimentos finalizados.\n");
  return 0;
}
