import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def process_and_plot(chosen_file_name):
    TEST_DIR = "./testes"
    INPUT_FILE = os.path.join(TEST_DIR, chosen_file_name)
    
    output_folder_name = os.path.splitext(chosen_file_name)[0]
    OUT_DIR = f"plots_{output_folder_name}"

    print("=====================================================")
    print(f"Processando arquivo: {INPUT_FILE}")
    print(f"Diretorio de saida:  {OUT_DIR}")
    print("=====================================================")

    if not os.path.exists(INPUT_FILE):
        print(f"Erro: Arquivo '{INPUT_FILE}' nao encontrado.")
        print("Por favor, execute 'make -f Makefile.mak' e 'mpirun -np 6 ./main' primeiro.")
        return 

    data = []

    try:
        with open(INPUT_FILE, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Erro ao ler o arquivo {INPUT_FILE}: {e}")
        return

    line_regex = re.compile(
        r"\|\s*(\d+)\s*\|" +
        r"\s*([\w\s\(\)]+)\s*\|" +
        r"\s*([\d\.]+)\s*\|" +
        r"\s*([\d\.]+)\s*\|"
    )

    for line in lines:
        match = line_regex.search(line)
        
        if match:
            try:
                tamanho = int(match.group(1).strip())
                threads_str = match.group(2).strip()
                tempo = float(match.group(3).strip())
                gflops = float(match.group(4).strip())
                
                data.append({
                    'Tamanho': tamanho,
                    'Threads_str': threads_str,
                    'Tempo': tempo,
                    'GFLOPS': gflops
                })
            except Exception as e:
                print(f"Aviso: Ignorando linha mal formatada: {line.strip()} ({e})")

    if not data:
        print(f"Erro: Nenhum dado de performance encontrado em '{INPUT_FILE}'.")
        return

    df = pd.DataFrame(data)

    is_openmp = df['Threads_str'].str.match(r'^\d+$')
    df.loc[is_openmp, 'Threads_str'] = df['Threads_str'] + " (OpenMP)"


    df['Processos'] = pd.to_numeric(df['Threads_str'].str.extract(r'(\d+)')[0], errors='coerce')

    df_seq = df[df['Threads_str'] == '1 (Seq)'][['Tamanho', 'Tempo']].rename(columns={'Tempo': 'Tempo_Sequencial'})

    df_par = df[df['Processos'].notna() & (df['Threads_str'] != '1 (Seq)')].copy()

    df_plot = pd.merge(df_par, df_seq, on='Tamanho')

    df_plot['Speedup'] = df_plot['Tempo_Sequencial'] / df_plot['Tempo']
    df_plot['Eficiencia'] = df_plot['Speedup'] / df_plot['Processos']

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Dados lidos e processados de '{INPUT_FILE}'. Gerando plots em '{OUT_DIR}'...")

    sizes = sorted(df_plot['Tamanho'].unique())

    plt.figure(figsize=(10, 7))
    for N in sizes:
        sub = df_plot[df_plot['Tamanho'] == N].sort_values('Processos')
        plt.plot(sub['Processos'], sub['Tempo'], marker='o', linestyle='--', label=f'N={N}')
    plt.xlabel('Numero de Processos/Threads (P)')
    plt.ylabel('Tempo Paralelo (s)')
    plt.title('Tempo Paralelo (T_par) vs Processos/Threads')
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.savefig(os.path.join(OUT_DIR, 'tempo_paralelo.png'), dpi=200, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 7))
    for N in sizes:
        sub = df_plot[df_plot['Tamanho'] == N].sort_values('Processos')
        plt.plot(sub['Processos'], sub['Speedup'], marker='s', linestyle='--', label=f'N={N}')

    if not df_plot.empty:
        all_procs = df_plot['Processos'].dropna().unique()
        if all_procs.size > 0:
            maxP = int(all_procs.max())
            minP = int(all_procs.min())
            if minP < 1: minP = 1
            plt.plot([minP, maxP], [minP, maxP], 'k-', label='Speedup Ideal (y=x)')
            
    plt.xlabel('Numero de Processos/Threads (P)')
    plt.ylabel('Speedup (T_seq / T_par)')
    plt.title('Speedup vs Numero de Processos/Threads')
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.savefig(os.path.join(OUT_DIR, 'speedup.png'), dpi=200, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 7))
    for N in sizes:
        sub = df_plot[df_plot['Tamanho'] == N].sort_values('Processos')
        plt.plot(sub['Processos'], sub['Eficiencia'], marker='^', linestyle='--', label=f'N={N}')

    plt.axhline(y=1.0, color='k', linestyle='-', label='Eficiencia Ideal (100%)')
    plt.xlabel('Numero de Processos/Threads (P)')
    plt.ylabel('Eficiencia (Speedup / P)')
    plt.title('Eficiencia vs Numero de Processos/Threads')
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.ylim(0, 1.1)
    plt.savefig(os.path.join(OUT_DIR, 'eficiencia.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 7))
    try:
        colors = plt.cm.get_cmap('tab10', len(sizes))
    except:
        colors = plt.cm.get_cmap('viridis', len(sizes))

    bar_width = 0.15
    all_labels = sorted(df['Threads_str'].unique(), key=lambda x: (pd.to_numeric(x.split()[0], errors='coerce') if x != 'BLAS' else 99, x))
    x_ticks = np.arange(len(all_labels))

    for i, N in enumerate(sizes):
        sub_n = df[df['Tamanho'] == N].set_index('Threads_str').reindex(all_labels)
        plt.bar(x_ticks + i*bar_width, sub_n['GFLOPS'], width=bar_width, label=f'N={N}', color=colors(i))

    plt.xlabel('Implementacao / Processos (P) / Threads (T)')
    plt.ylabel('GFLOPS (Bilioes de Operacoes/s)')
    plt.title('Desempenho (GFLOPS) por Implementacao (Maior e Melhor)')
    plt.xticks(x_ticks + bar_width * (len(sizes) - 1) / 2, all_labels, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, axis='y', linestyle=':')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'gflops_comparativo.png'), dpi=200, bbox_inches='tight')
    plt.close()


    print(f"Plots gerados com sucesso em: ./{OUT_DIR}/")

    print("\n--- Dados Processados (Speedup e Eficiencia recalculados) ---")
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    if not df_plot.empty:
        print(df_plot[['Tamanho', 'Processos', 'Threads_str', 'Tempo_Sequencial', 'Tempo', 'Speedup', 'Eficiencia']].to_string(index=False))
    else:
        print("Nenhum dado paralelo para calcular Speedup/Eficiencia (apenas 1 (Seq) ou BLAS).")

if __name__ == "__main__":
    log_principal = "teste7_par_mpi.txt"
    
    process_and_plot(log_principal)

    print("\nProcessamento concluido.")