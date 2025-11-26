#include "funcoes.h"
#include <sys/time.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32
#define ALPHA 1.0
#define BETA 0.0
#define TOLERANCE 1e-8
#define EPSILON 1e-12

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
    
    double *A = NULL;
    double *B = NULL;
    double *C = NULL;
    double *C_correto = NULL;
    
    double tempo_seq = 0.0;

    A = aloca_matriz(tam_matriz);
    B = aloca_matriz(tam_matriz);
    C_correto = aloca_matriz(tam_matriz); 
    C_naive = aloca_matriz(tam_matriz);
    C_tiled = aloca_matriz(tam_matriz);

    // inicializa_matrizes(A, B, tam_matriz);

    dgemm_cuda(A, B, C, tam_matriz);

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

  int row = blockIdx.y * BLOCK_SIZE + ty;
  int col = blockIdx.x * BLOCK_SIZE + tx;

  double cValue = 0.0;

  __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];

  int numTiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

  for (int m = 0; m < numTiles; ++m) {
    int k = m * BLOCK_SIZE + tx;
    As[ty][tx] = (row < N && k < N) ? A[row*N + k] : 0.0;
    k = m * BLOCK_SIZE + ty;
    Bs[ty][tx] = (k < N && col < N) ? B[k*N + col] : 0.0;

    __syncthreads();

    for (int k2 = 0; k2 < BLOCK_SIZE; ++k2)
      cValue += As[ty][k2] * Bs[k2][tx];

    __syncthreads();
  }

  if (row < N && col < N)
    C[row*N + col] = alpha * cValue + beta * C[row*N + col];
}

// --- Funções auxiliares do CUDA ---

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

void dgemm_cuda(double *A, double *B, double *C, int N) {
  size_t size_bytes = N * N * sizeof(double);

  double *d_A, *d_B, *d_C;
  cudaCheckError(cudaMalloc((void**)&d_A, size_bytes));
  cudaCheckError(cudaMalloc((void**)&d_B, size_bytes));
  cudaCheckError(cudaMalloc((void**)&d_C, size_bytes));

  cudaCheckError(cudaMemcpy(d_A, A, size_bytes, cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(d_B, B, size_bytes, cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemset(d_C, 0, size_bytes));

  dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 blocksPerGrid((N + BLOCK_SIZE - 1)/BLOCK_SIZE, (N + BLOCK_SIZE - 1)/BLOCK_SIZE);

  // --- CUDA Naive ---
  dgemm_naive_kernel<<<blocksPerGrid, threadsPerBlock>>>(N, ALPHA, d_A, d_B, BETA, d_C);
  cudaCheckError(cudaDeviceSynchronize());
  cudaCheckError(cudaMemcpy(C, d_C, size_bytes, cudaMemcpyDeviceToHost));

  // --- CUDA Tiled ---
  // Se quiser testar a versão Tiled, pode zerar C e chamar:
  // cudaCheckError(cudaMemset(d_C, 0, size_bytes));
  // dgemm_tiled_kernel<<<blocksPerGrid, threadsPerBlock>>>(N, ALPHA, d_A, d_B, BETA, d_C);
  // cudaCheckError(cudaDeviceSynchronize());
  // cudaCheckError(cudaMemcpy(C, d_C, size_bytes, cudaMemcpyDeviceToHost));

  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}
