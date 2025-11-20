import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines
import numpy as np
import matplotlib.colors as mcolors

def process_and_plot(chosen_file_name):
    TEST_DIR = "./testes"
    INPUT_FILE = os.path.join(TEST_DIR, chosen_file_name)
    
    output_folder_name = os.path.splitext(chosen_file_name)[0]
    OUT_DIR = f"plots/plots_{output_folder_name}"

    print("=====================================================")
    print(f"Processando arquivo: {INPUT_FILE}")
    print(f"Diretorio de saida:  {OUT_DIR}")
    print("=====================================================")

    if not os.path.exists(INPUT_FILE):
        print(f"Erro: Arquivo '{INPUT_FILE}' nao encontrado.")
        print("Por favor, execute 'make run NP=6' primeiro.")
        return 

    data = []

    try:
        with open(INPUT_FILE, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Erro ao ler o arquivo {INPUT_FILE}: {e}")
        return

    # Regex para capturar linhas da tabela
    line_regex = re.compile(
        r"\|\s*(\d+)\s*\|" +          # Tamanho
        r"\s*([\w\s\(\)]+)\s*\|" +    # Versao (Threads_str)
        r"\s*([\d\.]+)\s*\|" +        # Tempo
        r"\s*([\d\.]+)\s*\|"          # GFLOPS
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
            except:
                pass

    if not data:
        print(f"Erro: Nenhum dado encontrado em '{INPUT_FILE}'.")
        return

    df = pd.DataFrame(data)

    # Padroniza nomes
    is_just_number = df['Threads_str'].str.match(r'^\d+$')
    if is_just_number.any():
        df.loc[is_just_number, 'Threads_str'] = df['Threads_str'] + " (OpenMP)"

    # Extrai numero de processos
    df['Processos'] = pd.to_numeric(df['Threads_str'].str.extract(r'(\d+)')[0], errors='coerce')

    # Identifica Tipo (OpenMP ou MPI)
    def get_type(s):
        if "MPI" in s: return "MPI"
        if "OpenMP" in s: return "OpenMP"
        return "Outro"
    df['Tipo'] = df['Threads_str'].apply(get_type)

    # Separa dados Sequenciais e Paralelos
    df_seq = df[df['Threads_str'] == '1 (Seq)'][['Tamanho', 'Tempo']].rename(columns={'Tempo': 'Tempo_Sequencial'})
    df_par = df[df['Processos'].notna() & (df['Threads_str'] != '1 (Seq)')].copy()

    # Calcula metricas
    df_plot = pd.merge(df_par, df_seq, on='Tamanho')
    df_plot['Speedup'] = df_plot['Tempo_Sequencial'] / df_plot['Tempo']
    df_plot['Eficiencia'] = df_plot['Speedup'] / df_plot['Processos']

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Gerando gráficos unificados com símbolos distintos...")

    sizes = sorted(df_plot['Tamanho'].unique())

    # cria paleta contínua baseada nos valores de N
    norm = mcolors.Normalize(vmin=min(sizes), vmax=max(sizes))
    cmap = cm.get_cmap("tab10_r")  # pode usar plasma, inferno, turbo, magma...

    colors = {N: cmap(norm(N)) for N in sizes}

    # --- FUNÇÃO AUXILIAR DE PLOTAGEM ---
    def plot_metric(metric_col, ylabel, title, filename, tipo, ideal_line=False):
        plt.figure(figsize=(10, 7))

        # Filtra dados apenas do tipo desejado
        df_sub = df_plot[df_plot["Tipo"] == tipo]

        for N in sizes:
            sub = df_sub[df_sub['Tamanho'] == N].sort_values('Processos')
            if sub.empty:
                continue

            c = colors[N]

            # Linha conectando pontos
            plt.plot(sub['Processos'], sub[metric_col],
                     color=c, linestyle='-', linewidth=2, alpha=0.8)

            # Pontos
            marker = 'o' if tipo == "OpenMP" else 's'
            plt.scatter(sub['Processos'], sub[metric_col],
                        color=c, marker=marker, s=60, edgecolors='k')

        # Linha ideal
        if ideal_line and not df_sub.empty:
            all_procs = df_sub['Processos'].dropna().unique()
            if len(all_procs) > 0:
                mn, mx = int(all_procs.min()), int(all_procs.max())
                if metric_col == 'Speedup':
                    plt.plot([mn, mx], [mn, mx], 'k:', alpha=0.4, label='Ideal')
                elif metric_col == 'Eficiencia':
                    plt.axhline(1.0, color='k', linestyle=':', alpha=0.4, label='Ideal')

        plt.xlabel('Número de Threads/Processos (P)')
        plt.ylabel(ylabel)
        plt.title(f"{title} — {tipo}")
        plt.grid(True, linestyle=':', alpha=0.7)

        # Legenda das cores (N)
        color_handles = [
            mlines.Line2D([], [], color=colors[N], linestyle='-', label=f"N={N}")
            for N in sizes
        ]
        plt.legend(handles=color_handles, title="Tamanho", loc='upper left')

        if metric_col == 'Eficiencia':
            plt.ylim(0, 1.2)

        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, filename), dpi=200)
        plt.close()


    # --- GRÁFICOS DE LINHA ---

    # OPENMP
    plot_metric('Tempo', 'Tempo (s)', 'Tempo de Execução (OpenMP)', 'tempo_openmp.png', tipo="OpenMP")
    plot_metric('Speedup', 'Speedup', 'Speedup (OpenMP)', 'speedup_openmp.png', tipo="OpenMP", ideal_line=True)
    plot_metric(
        'Eficiencia', 'Eficiência (Speedup / P)', 'Eficiência (OpenMP)',
        'eficiencia_openmp.png', tipo="OpenMP", ideal_line=True)

    # MPI
    plot_metric('Tempo', 'Tempo (s)', 'Tempo de Execução (MPI)', 'tempo_mpi.png', tipo="MPI")
    plot_metric('Speedup', 'Speedup', 'Speedup (MPI)', 'speedup_mpi.png', tipo="MPI", ideal_line=True)
    plot_metric(
        'Eficiencia', 'Eficiência (Speedup / P)', 'Eficiência (MPI)',
        'eficiencia_mpi.png', tipo="MPI", ideal_line=True)

    # --- GRÁFICO DE BARRAS (GFLOPS) ---

    plt.figure(figsize=(12, 7))
    
    def sort_key(x):
        num_part = x.split()[0]
        if x == 'BLAS': return (999, 2)
        if num_part.isdigit():
            num = int(num_part)
            prio = 0 if "OpenMP" in x else (1 if "MPI" in x else 0)
            return (num, prio)
        return (999, 2)
    
    all_labels = sorted(df['Threads_str'].unique(), key=sort_key)
    x_ticks = np.arange(len(all_labels))
    bar_width = 0.15

    for i, N in enumerate(sizes):
        sub_n = df[df['Tamanho'] == N].set_index('Threads_str').reindex(all_labels)
        plt.bar(x_ticks + i*bar_width, sub_n['GFLOPS'], width=bar_width, label=f'N={N}', color=colors[N])

    plt.xlabel('Implementação')
    plt.ylabel('GFLOPS')
    plt.title('Desempenho (GFLOPS) Comparativo')
    plt.xticks(x_ticks + bar_width * (len(sizes) - 1) / 2, all_labels, rotation=45, ha='right')
    plt.legend(title="Tamanho")
    plt.grid(True, axis='y', linestyle=':')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'gflops_comparativo.png'), dpi=200)
    plt.close()

    print(f"Plots gerados com sucesso em: ./{OUT_DIR}/")
    
    if not df_plot.empty:
        print("\n--- Resumo dos Dados ---")
        print(df_plot[['Tamanho', 'Threads_str', 'Tempo', 'Speedup', 'Eficiencia']].to_string(index=False))

if __name__ == "__main__":
    log_principal = "teste8_cuda.txt" 
    process_and_plot(log_principal)
    print("\nConcluido.")