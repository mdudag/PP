#include "funcoes.h"
#include <cuda_runtime.h>

#define BSIZE 32
#define ALPHA 1.0
#define BETA 0.0
#define TOLERANCE 1e-8
#define EPSILON 1e-12

// Checagem de erros CUDA
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

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
  fprintf(file, "                  DGEMM CUDA (Naive e Tiled)                   \n");
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
    
    // Host
    double *h_A = NULL;
    double *h_B = NULL;
    double *h_C = NULL;
    double *h_C_Correto = NULL; // Sequencial
    
    double tempo_seq_cpu = 0.0;

    h_A = aloca_matriz(tam_matriz);
    h_B = aloca_matriz(tam_matriz);
    h_C = aloca_matriz(tam_matriz); 
    h_C_Correto = aloca_matriz(tam_matriz);

    inicializa_matrizes(h_A, h_B, tam_matriz);
    zera_matriz(h_C, tam_matriz);
    zera_matriz(h_C_Correto, tam_matriz);

    // --- Versao sequencial ---
    tempo_seq_cpu = medir_tempo_execucao(dgemm_sequencial, NUM_REPETICOES, h_A, h_B, h_C_Correto, tam_matriz);
    registrar_resultado(file, tam_matriz, "1 (Seq)", tempo_seq_cpu, tempo_seq_cpu, 1, 0.0);

    // --- Versao CUDA ---
    
    size_t size_bytes = tam_matriz * tam_matriz * sizeof(double);

    // Aloca memoria para matrizes na gpu 
    double *d_A, *d_B, *d_C;
    cudaCheckError(cudaMalloc((void**)&d_A, size_bytes));
    cudaCheckError(cudaMalloc((void**)&d_B, size_bytes));
    cudaCheckError(cudaMalloc((void**)&d_C, size_bytes));

    // Copia os valores das matrizes da CPU para as matrizes da GPU
    cudaCheckError(cudaMemcpy(d_A, h_A, size_bytes, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_B, h_B, size_bytes, cudaMemcpyHostToDevice));
    
    dim3 threadsPerBlock(BSIZE, BSIZE);
    dim3 blocksPerGrid((tam_matriz + BSIZE - 1)/BSIZE, (tam_matriz + BSIZE - 1)/BSIZE);
    
    // --- CUDA Naive ---

    printf(" Executando CUDA Naive...       "); fflush(stdout);
    cudaCheckError(cudaMemset(d_C, 0, size_bytes));

    dgemm_naive_kernel<<<blocksPerGrid, threadsPerBlock>>>(tam_matriz, ALPHA, d_A, d_B, BETA, d_C);
    cudaDeviceSynchronize();

    double start, end;
    double tempo_gpu, erro, gflops;

    // Calculo do tempo
    start = get_time();
    dgemm_naive_kernel<<<blocksPerGrid, threadsPerBlock>>>(tam_matriz, ALPHA, d_A, d_B, BETA, d_C);
    cudaCheckError(cudaDeviceSynchronize());
    end = get_time();

    // Metricas
    tempo_gpu = end - start; // Calculo do tempo
    double gflops_naive = (2.0 * tam_matriz * tam_matriz * tam_matriz * 1e-9) / tempo_gpu;
    
    // tempo = medir_tempo_execucao(dgemm_paralelo_openMP, NUM_REPETICOES, A, B, C, tam_matriz);
    // erro = valida_resultado(C_correto, C, tam_matriz);
    // int n_th = ? numero de nucleos
    // char nome_versao[20];
    // sprintf(nome_versao, "%d (CUDA)", n_th);
    // registrar_resultado(file, tam_matriz, nome_versao, tempo, tempo_seq_cpu, n_th, erro);
    
    cudaCheckError(cudaMemcpy(h_C, d_C, size_bytes, cudaMemcpyDeviceToHost));
    fprintf(file, "Tempo: %.4f s | GFLOPS: %.2f | Speedup: %.2fx\n", 
            tempo_gpu, gflops_naive, tempo_seq_cpu / tempo_gpu);
    
    validate_results(tam_matriz, h_C_Correto, h_C);

    // --- CUDA Tiled ---

    zera_matriz(h_C, tam_matriz);
    printf("   Executando CUDA Tiled...       "); fflush(stdout);
    cudaCheckError(cudaMemset(d_C, 0, size_bytes));

    // Calculo do tempo
    start = get_time();
    dgemm_tiled_kernel<<<blocksPerGrid, threadsPerBlock>>>(tam_matriz, ALPHA, d_A, d_B, BETA, d_C);
    cudaCheckError(cudaDeviceSynchronize());
    double end = get_time();

    // Metricas
    tempo_gpu = end - start;
    double gflops_tiled = (2.0 * tam_matriz * tam_matriz * tam_matriz * 1e-9) / tempo_gpu;

    cudaCheckError(cudaMemcpy(h_C, d_C, size_bytes, cudaMemcpyDeviceToHost));
    fprintf(file, "Tempo: %.4f s | GFLOPS: %.2f | Speedup: %.2fx\n", 
            tempo_gpu, gflops_tiled, tempo_seq_cpu / tempo_gpu);

    validate_results(tam_matriz, h_C_Correto, h_C);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(A); free(B); free(C); free(C_correto); 
  }

  fprintf(file, "\n[LOG] Experimentos finalizados.\n");
  fclose(file);
  return 0;
}

// --- Kernels CUDA ---

__global__ void dgemm_naive_kernel(int N, double alpha, double *A, double *B, double beta, double *C) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < N && col < N) {
    double cValue = 0.0;
    for (int k = 0; k < N; ++k) {
        cValue += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = alpha * cValue + beta * C[row * N + col];
  }
}

__global__ void dgemm_tiled_kernel(int N, double alpha, double *A, double *B, double beta, double *C) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = blockIdx.y * BSIZE + ty;
  int col = blockIdx.x * BSIZE + tx;

  double cValue = 0.0;

  __shared__ double As[BSIZE][BSIZE];
  __shared__ double Bs[BSIZE][BSIZE];

  int numTiles = (N + BSIZE - 1) / BSIZE;

  for (int m = 0; m < numTiles; ++m) {
    int k = m * BSIZE + tx;
    As[ty][tx] = (row < N && k < N) ? A[row*N + k] : 0.0;
    k = m * BSIZE + ty;
    Bs[ty][tx] = (k < N && col < N) ? B[k*N + col] : 0.0;

    __syncthreads();

    for (int k2 = 0; k2 < BSIZE; ++k2)
      cValue += As[ty][k2] * Bs[k2][tx];

    __syncthreads();
  }

  if (row < N && col < N)
    C[row*N + col] = alpha * cValue + beta * C[row*N + col];
}

// --- Funcao auxiliar do CUDA ---

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}
