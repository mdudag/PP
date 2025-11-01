CC = mpicc

CFLAGS = -O3 -fopenmp -march=native -Wall

LIBS = -lopenblas -lm

SRCS = main.c funcoes.c
TARGET = main

all: $(TARGET)

$(TARGET): $(SRCS) funcoes.h
	@echo "Compilando programa hibrido (MPI + OpenMP + BLAS)..."
	$(CC) $(CFLAGS) -o $(TARGET) $(SRCS) $(LIBS)
	@echo "Compilacao concluida: $(TARGET)"

run: all
	@echo "Executando testes (Exemplo: 4 processos MPI, N=1024)..."
	mpirun -np 4 ./$(TARGET) 1024

clean:
	@echo "Limpando arquivos gerados..."
	rm -f $(TARGET)
	rm -f testes/teste7_par_mpi.txt
	rm -rf plots_mpi
	rm -f *.o

.PHONY: all run clean