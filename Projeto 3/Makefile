CC = mpicc

CFLAGS = -O3 -fopenmp -march=native -Wall

LIBS = -lopenblas -lm

SRCS = main.c funcoes.c
TARGET = main

# Variáveis padrão para rodar que podem ser alteradas na hora de chamar o make
NP ?= 6

all: $(TARGET)

$(TARGET): $(SRCS) funcoes.h
	@echo "Compilando programa hibrido (MPI + OpenMP + BLAS)..."
	$(CC) $(CFLAGS) -o $(TARGET) $(SRCS) $(LIBS)
	@echo "Compilacao concluida: $(TARGET)"

run: all
	@echo "Executando bateria de testes (512 a 4096)..."
	@mkdir -p testes
	export OPENBLAS_NUM_THREADS=1 && mpirun -np $(NP) ./$(TARGET)

clean:
	@echo "Limpando arquivos gerados..."
	rm -f $(TARGET)
	rm -f testes/teste7_par_mpi.txt
	rm -rf plots_mpi
	rm -f *.o

.PHONY: all run clean