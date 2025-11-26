# =======================================
#  Makefile – Projeto 3: DGEMM com CUDA
# =======================================

# ------------- VERSOES ANTERIORES -------------

# Comando de compilação
CC = mpicc																# Compilador MPI
CFLAGS = -O3 -fopenmp -march=native -Wall # Flags de compilação
LIBS = -lopenblas -lm											# Bibliotecas

# Aquivos fonte e executável
SRCS = main.c funcoes.c
TARGET = main

# Variável padrão do número de núcleos, que podem ser alteradas na hora de chamar o make
NP ?= 6

# Compilação das versões anteriores
all: $(TARGET)

$(TARGET): $(SRCS) funcoes.h
	@echo "Compilando programa hibrido (MPI + OpenMP + BLAS)..."
	$(CC) $(CFLAGS) -o $(TARGET) $(SRCS) $(LIBS)
	@echo "Compilacao versões anteriores concluida: $(TARGET)"

# Compilação e execução das versões anteriores
run: all
	@echo "Executando bateria de testes das versoes anteriores(512 a 4096)..."
	@mkdir -p testes
	export OPENBLAS_NUM_THREADS=1 && mpirun -np $(NP) ./$(TARGET)
	@echo "Fim da execucao das versoes anteriores"

# ------------- VERSÃO CUDA -------------

# Comando de compilação
NVCC = nvcc											# Compilador CUDA
NVCC_FLAGS = -O3 -arch=$(ARCH)	# Flags do CUDA
ARCH = sm_60	# Arquitetura da GPU (Ampere)

# Aquivos fonte e executável CUDA
CUDA_SRCS = dgemm_cuda.cu funcoes.c
CUDA_TARGET = dgemm_cuda

# Pasta de saída do CUDA
BIN_DIR = bin

# Compilação CUDA
cuda: $(BIN_DIR)/$(CUDA_TARGET)

$(BIN_DIR)/$(CUDA_TARGET): $(CUDA_SRCS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $(BIN_DIR)/$(CUDA_TARGET) $(CUDA_SRCS)
	@echo "Compilacao CUDA concluida: $(BIN_DIR)/$(CUDA_TARGET)"

# Compilação e execução CUDA
run_cuda: cuda
	@echo "Executando bateria de testes CUDA (512 a 4096)..."
	./$(BIN_DIR)/$(CUDA_TARGET)
	@echo "Fim da execucao CUDA"

# ------------- LIMPEZA -------------

clean:
	@echo "Limpando arquivos gerados..."
	rm -f $(CUDA_TARGET) $(TARGET) *.o
	rm -rf $(BIN_DIR) testes plots
	@echo "Limpeza concluida."

.PHONY: all run cuda run_cuda clean