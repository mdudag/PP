#include "funcoes.h"

int main() {
  // Inicia arquivo
  FILE *file = NULL;
  char nomeFile[100] = "./testes/teste8_cuda.txt";

  // Continua a escrita no mesmo arquivo
  file = fopen(nomeFile, "a");
  if (!file) {
    printf("Erro ao abrir o arquivo %s\n", nomeFile);
    return 1;
  }

  fprintf(file, "----------------------------------------------------------------\n");
  fprintf(file, "                          DGEMM CUDA                            \n");
  fprintf(file, "----------------------------------------------------------------\n");

  fprintf(file, "\n--- INICIO DOS EXPERIMENTOS (Media de %d execucoes) ---\n", NUM_REPETICOES);
  fprintf(file, "| %-8s | %-12s | %-12s | %-10s | %-10s | %-10s | %-12s | %-9s |\n",
          "Tamanho", "Versao", "Tempo (s)", "GFLOPS", "Speedup", "Eficiencia", "Diff. Max", "Validacao");
  fprintf(file, "|----------|--------------|--------------|------------|------------|------------|--------------|-----------|\n");

  for (int idx = 0; idx < num_tam; idx++) {
    int tam_matriz = tam_matrizes[idx];

    printf("--> Testes para matriz %dx%d...\n", tam_matriz, tam_matriz);
    if (tam_matriz == 4096) printf("    (Isso pode demorar alguns minutos, aguarde...)\n");
    fflush(stdout); // Força a escrita na tela imediatamente
    
    double *A = NULL;
    double *B = NULL;
    double *C = NULL;
    double *C_correto = NULL; 
    
    double tempo_seq = 0.0;

    A = aloca_matriz(tam_matriz);
    B = aloca_matriz(tam_matriz);
    C = aloca_matriz(tam_matriz);
    C_correto = aloca_matriz(tam_matriz); 

    inicializa_matrizes(A, B, C, tam_matriz);

    dgemm_cuda(A, B, C, tam_matriz);

    free(A); free(B); free(C); free(C_correto); 
  }

  fprintf(file, "\n[LOG] Experimentos finalizados.\n");
  fclose(file);
  return 0;
}
