import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.fft import fft, fftfreq
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import seaborn as sns

# -----------------------------
# Funções de métricas
def calcular_metricas(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    total = cm.sum()
    sensibilidade = []
    especificidade = []
    for i in range(len(cm)):
        vp = cm[i, i]
        fn = cm[i, :].sum() - vp
        fp = cm[:, i].sum() - vp
        vn = total - (vp + fn + fp)
        sensibilidade.append(vp / (vp + fn) if (vp + fn) > 0 else 0)
        especificidade.append(vn / (vn + fp) if (vn + fp) > 0 else 0)

    prec, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0, labels=range(len(class_names)))

    return {
        "Acurácia": np.trace(cm) / total if total > 0 else 0,
        "Sensibilidade média": np.mean(sensibilidade),
        "Especificidade média": np.mean(especificidade),
        "F1-score médio": np.mean(f1),
        "Matriz de confusão": cm
    }

# -----------------------------
# Função para extrair features via FFT
def extract_features(segment, fs):
    N = len(segment)
    yf = fft(segment)
    mag = 2.0 / N * np.abs(yf[0:N//2])
    return mag

# -----------------------------
# 1. Carregar e preparar dataset de vibrações
root_dir = ''  # Ajuste o caminho se necessário
classes = {'teste1': 0, 'teste2': 1, 'teste3': 2}  # Incluindo tipo2
class_names = list(classes.keys())
fs_fixed = 100  # Taxa de amostragem fixa de 100 Hz

X_list = []
y_list = []

for tipo, label in classes.items():
    folder = os.path.join(root_dir, tipo)
    if not os.path.exists(folder):
        print(f"Pasta {folder} não encontrada. Ignorando...")
        continue
    for filename in os.listdir(folder):
        if filename.endswith('.csv'):
            filepath = os.path.join(folder, filename)
            try:
                df = pd.read_csv(filepath)
                df.columns = df.columns.str.strip()  # remover espaços

                # Estrutura 1: colunas AccelerationX/Y/Z
                if {'AccelerationX', 'AccelerationY', 'AccelerationZ'}.issubset(df.columns):
                    signal = np.sqrt(
                        df['AccelerationX']**2 +
                        df['AccelerationY']**2 +
                        df['AccelerationZ']**2
                    ).values
                    time_col = 'SamplingTime'

                # Estrutura 2: colunas x, y, z (+ seconds_elapsed, time)
                elif {'x', 'y', 'z'}.issubset(df.columns):
                    signal = np.sqrt(
                        df['x']**2 +
                        df['y']**2 +
                        df['z']**2
                    ).values
                    # usar seconds_elapsed se existir, senão time
                    if 'seconds_elapsed' in df.columns:
                        time_col = 'seconds_elapsed'
                    elif 'time' in df.columns:
                        time_col = 'time'
                    else:
                        print(f"Arquivo {filename}: nenhuma coluna de tempo encontrada, usando índice")
                        df['index_time'] = np.arange(len(df))
                        time_col = 'index_time'

                else:
                    print(f"Erro no arquivo {filename}: Colunas de aceleração não encontradas. Colunas disponíveis: {df.columns.tolist()}")
                    continue

                # verificar tempo
                if time_col not in df.columns:
                    print(f"Erro no arquivo {filename}: Coluna {time_col} não encontrada.")
                    continue
                time = df[time_col].values
                print(f"Arquivo {filename} ({tipo}): Primeiros 5 valores de {time_col}: {time[:5]}")

                # taxa de amostragem fixa
                fs = fs_fixed
                print(f"Arquivo {filename} ({tipo}): Usando taxa de amostragem fixa {fs} Hz")

                # normalizar o sinal
                signal = (signal - np.mean(signal)) / np.std(signal) if np.std(signal) > 0 else signal

                # segmentação em 1s
                segment_length = int(fs * 1)
                num_segments = len(signal) // segment_length
                if num_segments == 0:
                    print(f"Sinal muito curto ({len(signal)} amostras). Pulando {filename}")
                    continue

                for i in range(num_segments):
                    start, end = i * segment_length, (i + 1) * segment_length
                    segment = signal[start:end]

                    # extrair features FFT
                    features = extract_features(segment, fs)
                    X_list.append(features)
                    y_list.append(label)

            except Exception as e:
                print(f"Erro ao processar {filename} ({tipo}): {str(e)}")
                continue

X = np.array(X_list)
y = np.array(y_list)

# Verificar se há dados suficientes
if len(X) == 0 or len(y) == 0:
    raise ValueError("Nenhum dado válido foi carregado. Verifique os arquivos e pastas.")
unique_classes = np.unique(y)
if len(unique_classes) < 2:
    raise ValueError(f"Apenas uma classe foi carregada ({unique_classes}). São necessárias pelo menos duas classes.")
elif len(unique_classes) < 3:
    print(f"Aviso: Apenas {len(unique_classes)} classes foram carregadas ({unique_classes}). Continuando com as classes disponíveis.")

# Verificar consistência do tamanho das features
max_len = max(len(f) for f in X)
X_padded = np.array([np.pad(f, (0, max_len - len(f)), mode='constant') for f in X])

# -----------------------------
# 2. Divisão treino/validação/teste
X_train, X_temp, y_train, y_temp = train_test_split(
    X_padded, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Normalização das features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# -----------------------------
# 3. Modelos e grids de hiperparâmetros
param_grids = {
    "KNN": (KNeighborsClassifier(), {"n_neighbors": [3, 5, 7, 9]}),
    "Árvore de Decisão": (DecisionTreeClassifier(random_state=42),
                          {"max_depth": [3, 5, 7, 9]}),
    "Floresta Aleatória": (RandomForestClassifier(random_state=42),
                           {"n_estimators": [5, 10, 50, 100],
                            "max_depth": [3, 5, 7, 9]}),
    "SVM": (SVC(random_state=42),
            {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}),
    "MLP": (MLPClassifier(max_iter=500, random_state=42),
            {"hidden_layer_sizes": [(64,), (128,), (64,64), (128,128)],
             "activation": ["relu", "tanh"]}),
    "Naive Bayes": (GaussianNB(), {"var_smoothing": [1e-9, 1e-8, 1e-7]})
}

# -----------------------------
# 4. Treino com validação
best_models = {}
for name, (model, grid) in param_grids.items():
    try:
        grid_search = GridSearchCV(model, grid, cv=3, n_jobs=-1, error_score='raise')
        grid_search.fit(X_train, y_train)
        best_models[name] = grid_search.best_estimator_
        print(f"\n=== {name} ===")
        print("Melhor configuração encontrada:", grid_search.best_params_)
        y_val_pred = best_models[name].predict(X_val)
        metricas_val = calcular_metricas(y_val, y_val_pred, class_names)
        print("Métricas de Validação:")
        for k, v in metricas_val.items():
            if k != "Matriz de confusão":
                print(f"{k}: {v:.4f}")
    except Exception as e:
        print(f"Erro ao treinar {name}: {str(e)}")
        continue

# -----------------------------
# 5. Avaliação final no teste
print("\n########## RESULTADOS FINAIS (TESTE) ##########")
for name, model in best_models.items():
    y_test_pred = model.predict(X_test)
    metricas_test = calcular_metricas(y_test, y_test_pred, class_names)
    print(f"\n=== {name} ===")
    for k, v in metricas_test.items():
        if k != "Matriz de confusão":
            print(f"{k}: {v:.4f}")
    cm = metricas_test["Matriz de confusão"]
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(f"Matriz de Confusão - {name} (Teste)")
    plt.xlabel('Classe Prevista')
    plt.ylabel('Classe Verdadeira')
    plt.show()

# -----------------------------
# 6. Salvando resultados para o relatório
with open('relatorio_resultados.txt', 'w') as f:
    f.write("Resultados da Avaliação dos Modelos\n\n")
    for name, model in best_models.items():
        y_test_pred = model.predict(X_test)
        metricas_test = calcular_metricas(y_test, y_test_pred, class_names)
        f.write(f"=== {name} ===\n")
        for k, v in metricas_test.items():
            if k != "Matriz de confusão":
                f.write(f"{k}: {v:.4f}\n")
        f.write("\nMatriz de Confusão:\n")
        f.write(f"{metricas_test['Matriz de confusão']}\n\n")