"""
Requisitos:
  uv add numpy pandas scikit-learn matplotlib seaborn ucimlrepo openpyxl
"""

import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate, learning_curve
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss,
    classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

# diretório de saída
BASE_DIR = Path.cwd()
OUT_DIR = BASE_DIR / "mlp_back_propagation" / "resultados_mlp"
OUT_DIR.mkdir(exist_ok=True)

# estilo global dos gráficos
plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261", "#264653", "#A8DADC"]
SEEDS   = [0, 42, 7, 123, 999]          # 5 seeds para Q1
CLASSES = ["Seker","Barbunya","Bombay","Cali","Dermason","Horoz","Sira"]


# UTILITÁRIOS
def carregar_dados():
    """Carrega o Dry Bean Dataset do UCI (requer ucimlrepo) ou via fallback."""
    try:
        from ucimlrepo import fetch_ucirepo
        print("  Baixando Dry Bean Dataset do UCI ML Repository...")
        ds = fetch_ucirepo(id=602)
        X = ds.data.features
        y = ds.data.targets.squeeze()
    except Exception:
        print("  ucimlrepo indisponível — usando sklearn fetch_openml como fallback...")
        from sklearn.datasets import fetch_openml
        ds = fetch_openml(name="DryBeanDataset", version=1, as_frame=True, parser="auto")
        X = ds.data
        y = ds.target

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print(f"  Dataset carregado: {X.shape[0]} exemplos, {X.shape[1]} atributos, "
          f"{len(np.unique(y_enc))} classes")
    print(f"  Classes: {list(le.classes_)}")
    return X.values.astype(float), y_enc, list(le.classes_)


def salvar_fig(fig, nome):
    caminho = OUT_DIR / nome
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Gráfico salvo: {caminho}")


def tabela_resultados(df, nome_arquivo):
    caminho = OUT_DIR / nome_arquivo
    df.to_csv(caminho, index=False)
    print(f"  Tabela salva: {caminho}")


def treinar_e_monitorar(mlp, X_train, y_train, X_val=None, y_val=None):
    """
    Treina época a época e coleta loss/acurácia.
    Retorna dicionário com histórico.
    """
    hist = {"train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   []}

    mlp.set_params(warm_start=True, max_iter=1)
    for _ in range(mlp.max_iter if hasattr(mlp, "_max_iter_orig") else 100):
        mlp.fit(X_train, y_train)
        prob_tr = mlp.predict_proba(X_train)
        hist["train_loss"].append(log_loss(y_train, prob_tr))
        hist["train_acc"].append(accuracy_score(y_train, mlp.predict(X_train)))
        if X_val is not None:
            prob_v = mlp.predict_proba(X_val)
            hist["val_loss"].append(log_loss(y_val, prob_v))
            hist["val_acc"].append(accuracy_score(y_val, mlp.predict(X_val)))
    mlp.set_params(warm_start=False)
    return hist


# Q1 – EXPLORAÇÃO INICIAL: PESOS INICIAIS E FUNÇÕES DE ATIVAÇÃO
def questao_1(X_train, y_train):
    """
    5 treinamentos com seeds diferentes.
    Arquitetura fixa: 2 camadas ocultas [128, 64], ReLU, Softmax implícita.
    Otimizador: Adam  |  LR: 0.001  |  Épocas: 100
    """
    print("\n" + "="*70)
    print("Q1 – EXPLORAÇÃO INICIAL: PESOS INICIAIS E FUNÇÕES DE ATIVAÇÃO")
    print("="*70)

    EPOCAS   = 100
    LR       = 0.001
    TOPOLOGY = (128, 64)

    registros   = []
    historicos  = []

    for i, seed in enumerate(SEEDS):
        print(f"  Treinamento {i+1}/5  (seed={seed}) ...", end=" ", flush=True)
        t0 = time.time()
        mlp = MLPClassifier(
            hidden_layer_sizes=TOPOLOGY,
            activation="relu",
            solver="adam",
            learning_rate_init=LR,
            max_iter=EPOCAS,
            random_state=seed,
            early_stopping=False,
            verbose=False
        )
        mlp.fit(X_train, y_train)
        elapsed = time.time() - t0

        loss_final = mlp.loss_
        acc_final  = accuracy_score(y_train, mlp.predict(X_train))
        f1_final   = f1_score(y_train, mlp.predict(X_train), average="macro")

        registros.append({
            "Treino": i+1, "Seed": seed,
            "Loss Final": round(loss_final, 4),
            "Acurácia Final": round(acc_final, 4),
            "F1-Macro": round(f1_final, 4),
            "Épocas Reais": len(mlp.loss_curve_),
            "Tempo (s)": round(elapsed, 2)
        })
        historicos.append(mlp.loss_curve_)
        print(f"loss={loss_final:.4f}  acc={acc_final:.4f}")

    df = pd.DataFrame(registros)
    print("\n  Resultados Q1:")
    print(df.to_string(index=False))
    tabela_resultados(df, "q1_resultados.csv")

    # Gráficos 
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Q1 – Curvas de Convergência por Inicialização", fontsize=14, fontweight="bold")

    for i, (curve, seed) in enumerate(zip(historicos, SEEDS)):
        axes[0].plot(curve, color=PALETTE[i], label=f"Seed {seed}", alpha=0.85)
    axes[0].set_title("Perda (Loss) por Época"); axes[0].set_xlabel("Época"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].set_yscale("log")

    bar_x = [f"S{s}" for s in SEEDS]
    bar_y = df["Acurácia Final"].values
    bars  = axes[1].bar(bar_x, bar_y, color=PALETTE[:5], edgecolor="white")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Acurácia Final por Inicialização")
    axes[1].set_ylabel("Acurácia")
    for bar, v in zip(bars, bar_y):
        axes[1].text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.3f}",
                     ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    salvar_fig(fig, "q1_curvas_convergencia.png")

    print("\n  Análise Q1:")
    print(f"  • Acurácia média  : {df['Acurácia Final'].mean():.4f}")
    print(f"  • Acurácia std    : {df['Acurácia Final'].std():.4f}")
    print(f"  • F1-macro médio  : {df['F1-Macro'].mean():.4f}")
    std_acc = df["Acurácia Final"].std()
    if std_acc < 0.01:
        print("  • Conclusão: treinamento ESTÁVEL — baixa sensibilidade à inicialização.")
    else:
        print("  • Conclusão: sensibilidade MODERADA à inicialização dos pesos.")

    return TOPOLOGY, LR, "adam"


# Q2 – HIPERPARÂMETROS: TAXA DE APRENDIZADO E TERMO MOMENTO
def questao_2(X_train, y_train, topology):
    """
    Grid Search: lr ∈ {0.001, 0.01, 0.1, 0.5}  ×  momentum ∈ {0.5, 0.7, 0.9}
    Critério de parada: loss ≤ 0.001 ou 200 épocas
    Otimizador: SGD com momentum (para poder variar o momentum explicitamente)
    """
    print("\n" + "="*70)
    print("Q2 – HIPERPARÂMETROS: TAXA DE APRENDIZADO E TERMO MOMENTO")
    print("="*70)

    LRS      = [0.001, 0.01, 0.1, 0.5]
    MOMENTS  = [0.5, 0.7, 0.9]
    MAX_EP   = 200
    TOL_LOSS = 0.001

    registros  = []
    curvas     = {}

    for lr in LRS:
        for mo in MOMENTS:
            key = f"lr={lr}, mom={mo}"
            print(f"  {key} ...", end=" ", flush=True)
            mlp = MLPClassifier(
                hidden_layer_sizes=topology,
                activation="relu",
                solver="sgd",
                learning_rate_init=lr,
                momentum=mo,
                max_iter=1,
                warm_start=True,
                random_state=42,
                verbose=False,
                n_iter_no_change=MAX_EP
            )
            curve = []
            epocas_conv = MAX_EP
            t0 = time.time()
            for ep in range(MAX_EP):
                try:
                    mlp.fit(X_train, y_train)
                    loss_ep = mlp.loss_
                    curve.append(loss_ep)
                    if loss_ep <= TOL_LOSS:
                        epocas_conv = ep + 1
                        break
                except Exception:
                    curve.append(np.nan)
                    break
            elapsed = time.time() - t0

            loss_final = curve[-1] if curve else np.nan
            acc_final  = accuracy_score(y_train, mlp.predict(X_train))
            curvas[key] = curve

            registros.append({
                "LR": lr, "Momentum": mo,
                "Loss Final": round(loss_final, 6),
                "Acurácia": round(acc_final, 4),
                "Épocas p/ convergir": epocas_conv,
                "Tempo (s)": round(elapsed, 2),
                "Convergiu": loss_final <= TOL_LOSS
            })
            print(f"loss={loss_final:.6f}  épocas={epocas_conv}")

    df = pd.DataFrame(registros)
    print("\n  Resultados Q2:")
    print(df.to_string(index=False))
    tabela_resultados(df, "q2_resultados.csv")

    # ── Heatmap de épocas até convergência ────────────────────────────────
    pivot_ep  = df.pivot(index="LR", columns="Momentum", values="Épocas p/ convergir")
    pivot_acc = df.pivot(index="LR", columns="Momentum", values="Acurácia")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Q2 – Grid Search: LR × Momentum", fontsize=14, fontweight="bold")

    sns.heatmap(pivot_ep, annot=True, fmt=".0f", cmap="YlOrRd_r", ax=axes[0])
    axes[0].set_title("Épocas até Convergência (menos = melhor)")

    sns.heatmap(pivot_acc, annot=True, fmt=".3f", cmap="YlGn", ax=axes[1])
    axes[1].set_title("Acurácia Final")

    plt.tight_layout()
    salvar_fig(fig, "q2_heatmap_grid.png")

    # Curvas de convergência
    fig, axes = plt.subplots(len(MOMENTS), 1, figsize=(12, 4*len(MOMENTS)), sharex=True)
    fig.suptitle("Q2 – Curvas de Perda por LR (por Momentum)", fontsize=14, fontweight="bold")

    for ax, mo in zip(axes, MOMENTS):
        for j, lr in enumerate(LRS):
            key = f"lr={lr}, mom={mo}"
            if key in curvas:
                ax.plot(curvas[key], color=PALETTE[j], label=f"LR={lr}", alpha=0.85)
        ax.axhline(TOL_LOSS, ls="--", color="gray", lw=1, label="Critério (0.001)")
        ax.set_title(f"Momentum = {mo}"); ax.set_ylabel("Loss"); ax.legend(loc="upper right")
        ax.set_yscale("log")
    axes[-1].set_xlabel("Época")
    plt.tight_layout()
    salvar_fig(fig, "q2_curvas_convergencia.png")

    # Melhor combinação
    conv = df[df["Convergiu"] == True]
    if len(conv):
        best = conv.loc[conv["Épocas p/ convergir"].idxmin()]
    else:
        best = df.loc[df["Loss Final"].idxmin()]

    best_lr  = best["LR"]
    best_mom = best["Momentum"]
    print(f"\n  Melhor combinação: LR={best_lr}, Momentum={best_mom}")
    print(f"  Loss={best['Loss Final']:.6f}  Acurácia={best['Acurácia']:.4f}  "
          f"Épocas={best['Épocas p/ convergir']}")

    return best_lr, best_mom


# Q3 – TOPOLOGIA: NÚMERO DE CAMADAS E NEURÔNIOS
def questao_3(X_train, y_train, X_val, y_val, lr, mom):
    """
    Varia camadas ocultas: 1, 2, 3
    Varia neurônios por camada: 10, 30, 50, 80, 100
    Critério de parada: loss ≤ 0.001 ou 150 épocas
    """
    print("\n" + "="*70)
    print("Q3 – TOPOLOGIA: NÚMERO DE CAMADAS E NEURÔNIOS POR CAMADA")
    print("="*70)

    CAMADAS  = [1, 2, 3]
    NEURONIOS= [10, 30, 50, 80, 100]
    MAX_EP   = 150
    TOL_LOSS = 0.001

    registros = []

    for n_layers in CAMADAS:
        for n_units in NEURONIOS:
            topo = tuple([n_units] * n_layers)
            label = f"{n_layers}x{n_units}"
            print(f"  Topologia {label} ...", end=" ", flush=True)

            mlp = MLPClassifier(
                hidden_layer_sizes=topo,
                activation="relu",
                solver="sgd",
                learning_rate_init=lr,
                momentum=mom,
                max_iter=1,
                warm_start=True,
                random_state=42,
                verbose=False
            )
            tr_losses, vl_losses = [], []
            epocas_conv = MAX_EP
            t0 = time.time()

            for ep in range(MAX_EP):
                try:
                    mlp.fit(X_train, y_train)
                    l_tr = log_loss(y_train, mlp.predict_proba(X_train))
                    l_vl = log_loss(y_val,   mlp.predict_proba(X_val))
                    tr_losses.append(l_tr)
                    vl_losses.append(l_vl)
                    if l_tr <= TOL_LOSS:
                        epocas_conv = ep + 1
                        break
                except Exception:
                    tr_losses.append(np.nan)
                    vl_losses.append(np.nan)
                    break

            elapsed = time.time() - t0
            acc_val = accuracy_score(y_val, mlp.predict(X_val))
            f1_val  = f1_score(y_val, mlp.predict(X_val), average="macro", zero_division=0)

            registros.append({
                "Topologia": label,
                "Camadas": n_layers,
                "Neurônios/Camada": n_units,
                "Loss Treino Final": round(tr_losses[-1] if tr_losses else 9999, 4),
                "Loss Val. Final":   round(vl_losses[-1] if vl_losses else 9999, 4),
                "Acurácia Val.":     round(acc_val, 4),
                "F1-Macro Val.":     round(f1_val, 4),
                "Épocas Conv.":      epocas_conv,
                "Tempo (s)":         round(elapsed, 2),
                "_tr_curve":         tr_losses,
                "_vl_curve":         vl_losses
            })
            print(f"loss_tr={tr_losses[-1]:.4f}  loss_vl={vl_losses[-1]:.4f}  "
                  f"f1={f1_val:.4f}  t={elapsed:.1f}s")

    df_full = pd.DataFrame(registros)
    df_show = df_full.drop(columns=["_tr_curve","_vl_curve"])
    print("\n  Resultados Q3:")
    print(df_show.to_string(index=False))
    tabela_resultados(df_show, "q3_resultados.csv")

    # ── Heatmap F1-Macro ──────────────────────────────────────────────────
    pivot_f1 = df_show.pivot(index="Camadas", columns="Neurônios/Camada", values="F1-Macro Val.")
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(pivot_f1, annot=True, fmt=".3f", cmap="Blues", ax=ax)
    ax.set_title("Q3 – F1-Macro Validação por Topologia", fontsize=13, fontweight="bold")
    plt.tight_layout()
    salvar_fig(fig, "q3_heatmap_f1.png")

    # Curvas das 4 melhores topologias
    top4 = df_full.nlargest(4, "F1-Macro Val.")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Q3 – Curvas de Perda: 4 Melhores Topologias", fontsize=14, fontweight="bold")

    for ax, (_, row) in zip(axes.flatten(), top4.iterrows()):
        ax.plot(row["_tr_curve"], color=PALETTE[0], label="Treino")
        ax.plot(row["_vl_curve"], color=PALETTE[1], label="Validação", ls="--")
        ax.axhline(TOL_LOSS, ls=":", color="gray", lw=1)
        ax.set_title(f"Topologia {row['Topologia']}  (F1={row['F1-Macro Val.']:.3f})")
        ax.set_xlabel("Época"); ax.set_ylabel("Loss"); ax.legend()

    plt.tight_layout()
    salvar_fig(fig, "q3_top4_curvas.png")

    melhores = top4["Topologia"].tolist()
    melhores_rows = top4[["Topologia","Camadas","Neurônios/Camada",
                           "F1-Macro Val.","Loss Val. Final","Tempo (s)"]].copy()
    print(f"\n  4 melhores topologias: {melhores}")
    tabela_resultados(melhores_rows, "q3_top4.csv")

    best_topo = tuple([top4.iloc[0]["Neurônios/Camada"]] * int(top4.iloc[0]["Camadas"]))
    return best_topo, top4


# Q4 – INFLUÊNCIA DOS DADOS DE TREINAMENTO
def questao_4(X, y, best_topo, lr, mom):
    """
    Varia tamanho do conjunto de treinamento: 20%, 40%, 60%, 80%, 100%
    Divisão estratificada; avalia nos conjuntos de validação e teste.
    """
    print("\n" + "="*70)
    print("Q4 – INFLUÊNCIA DOS DADOS DE TREINAMENTO")
    print("="*70)

    # Separa teste fixo (20%)
    X_main, X_test, y_main, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    fracoes = [0.20, 0.40, 0.60, 0.80, 1.00]
    registros = []

    for frac in fracoes:
        if frac < 1.0:
            X_tr, _, y_tr, _ = train_test_split(
                X_main, y_main, train_size=frac, random_state=42, stratify=y_main
            )
        else:
            X_tr, y_tr = X_main, y_main

        X_tr2, X_val, y_tr2, y_val = train_test_split(
            X_tr, y_tr, test_size=0.15, random_state=42, stratify=y_tr
        )

        print(f"  Fração {int(frac*100)}% → {len(X_tr2)} treino ...", end=" ", flush=True)
        t0 = time.time()

        mlp = MLPClassifier(
            hidden_layer_sizes=best_topo,
            activation="relu",
            solver="sgd",
            learning_rate_init=lr,
            momentum=mom,
            max_iter=150,
            random_state=42,
            verbose=False
        )
        mlp.fit(X_tr2, y_tr2)
        elapsed = time.time() - t0

        acc_val  = accuracy_score(y_val,  mlp.predict(X_val))
        acc_test = accuracy_score(y_test, mlp.predict(X_test))
        f1_val   = f1_score(y_val,  mlp.predict(X_val),  average="macro", zero_division=0)
        f1_test  = f1_score(y_test, mlp.predict(X_test), average="macro", zero_division=0)
        loss_tr  = log_loss(y_tr2, mlp.predict_proba(X_tr2))
        loss_val = log_loss(y_val, mlp.predict_proba(X_val))

        registros.append({
            "Fração": f"{int(frac*100)}%",
            "N Treino": len(X_tr2),
            "Loss Treino": round(loss_tr, 4),
            "Loss Val.":   round(loss_val, 4),
            "Acc Val.":    round(acc_val, 4),
            "Acc Teste":   round(acc_test, 4),
            "F1 Val.":     round(f1_val, 4),
            "F1 Teste":    round(f1_test, 4),
            "Tempo (s)":   round(elapsed, 2)
        })
        print(f"f1_val={f1_val:.4f}  f1_test={f1_test:.4f}")

    df = pd.DataFrame(registros)
    print("\n  Resultados Q4:")
    print(df.to_string(index=False))
    tabela_resultados(df, "q4_resultados.csv")

    # Curvas de generalização
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Q4 – Curvas de Generalização vs. Tamanho do Conjunto", fontsize=13, fontweight="bold")

    for ax, metrica, titulo in zip(
        axes,
        [("F1 Val.", "F1 Teste"), ("Acc Val.", "Acc Teste"), ("Loss Val.", "Loss Treino")],
        ["F1-Macro", "Acurácia", "Loss"]
    ):
        m1, m2 = metrica
        ax.plot(df["Fração"], df[m1], marker="o", color=PALETTE[0], label=m1)
        ax.plot(df["Fração"], df[m2], marker="s", color=PALETTE[1], ls="--", label=m2)
        ax.set_title(titulo); ax.set_xlabel("Fração de Dados"); ax.set_ylabel(titulo)
        ax.legend(); ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    salvar_fig(fig, "q4_curvas_generalizacao.png")

    return X_main, X_test, y_main, y_test


# Q5 – INFLUÊNCIA DOS ATRIBUTOS
def questao_5(X_main, y_main, X_test, y_test, feature_names, best_topo, lr, mom):
    """
    Estratégias de seleção de atributos:
      A) Todos os 16 atributos
      B) Importância por permutação (top 8 e top 12)
      C) Remoção de atributos correlacionados (limiar r > 0.95)
      D) Atributos aleatórios (baseline)
    """
    print("\n" + "="*70)
    print("Q5 – INFLUÊNCIA DOS ATRIBUTOS")
    print("="*70)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_main, y_main, test_size=0.20, random_state=42, stratify=y_main
    )

    scaler = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_tst_s = scaler.transform(X_test)

    # Importância por permutação (modelo base)
    print("  Calculando importâncias por permutação ...", end=" ", flush=True)
    base_mlp = MLPClassifier(
        hidden_layer_sizes=best_topo, activation="relu",
        solver="sgd", learning_rate_init=lr, momentum=mom,
        max_iter=100, random_state=42
    )
    base_mlp.fit(X_tr_s, y_tr)
    perm = permutation_importance(base_mlp, X_val_s, y_val,
                                  n_repeats=5, random_state=42, scoring="f1_macro")
    imp_mean = perm.importances_mean
    idx_sorted = np.argsort(imp_mean)[::-1]
    print("ok")

    # Correlação 
    corr_matrix = np.corrcoef(X_tr_s.T)
    redundantes = set()
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            if abs(corr_matrix[i, j]) > 0.95:
                redundantes.add(j)
    idx_sem_corr = [i for i in range(len(feature_names)) if i not in redundantes]

    experimentos = {
        "Todos (16)":        list(range(len(feature_names))),
        "Top 12 permutação": idx_sorted[:12].tolist(),
        "Top 8 permutação":  idx_sorted[:8].tolist(),
        "Sem alta corr.":    idx_sem_corr,
        "Aleatório (5)":     np.random.RandomState(42).choice(
                                 len(feature_names), 5, replace=False).tolist()
    }

    registros = []
    historicos = {}

    for nome, cols in experimentos.items():
        print(f"  {nome} ({len(cols)} atributos) ...", end=" ", flush=True)
        t0 = time.time()
        mlp = MLPClassifier(
            hidden_layer_sizes=best_topo, activation="relu",
            solver="sgd", learning_rate_init=lr, momentum=mom,
            max_iter=1, warm_start=True, random_state=42
        )
        tr_l, vl_l = [], []
        for _ in range(150):
            try:
                mlp.fit(X_tr_s[:, cols], y_tr)
                tr_l.append(log_loss(y_tr, mlp.predict_proba(X_tr_s[:, cols])))
                vl_l.append(log_loss(y_val, mlp.predict_proba(X_val_s[:, cols])))
            except Exception:
                tr_l.append(np.nan); vl_l.append(np.nan)
                break

        elapsed = time.time() - t0
        f1_val  = f1_score(y_val,  mlp.predict(X_val_s[:, cols]),  average="macro", zero_division=0)
        f1_test = f1_score(y_test, mlp.predict(X_tst_s[:, cols]), average="macro", zero_division=0)
        acc_val = accuracy_score(y_val, mlp.predict(X_val_s[:, cols]))

        historicos[nome] = (tr_l, vl_l)
        registros.append({
            "Conjunto de Atributos": nome,
            "N Atributos": len(cols),
            "Loss Treino": round(tr_l[-1] if tr_l else 9999, 4),
            "Loss Val.":   round(vl_l[-1] if vl_l else 9999, 4),
            "Acc Val.":    round(acc_val, 4),
            "F1 Val.":     round(f1_val, 4),
            "F1 Teste":    round(f1_test, 4),
            "Tempo (s)":   round(elapsed, 2)
        })
        print(f"f1_val={f1_val:.4f}  f1_test={f1_test:.4f}")

    df = pd.DataFrame(registros)
    print("\n  Resultados Q5:")
    print(df.to_string(index=False))
    tabela_resultados(df, "q5_resultados.csv")

    # Gráfico de importância 
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Q5 – Análise de Atributos", fontsize=14, fontweight="bold")

    axes[0].barh(
        [feature_names[i] for i in idx_sorted],
        imp_mean[idx_sorted],
        color=PALETTE[2], edgecolor="white"
    )
    axes[0].set_title("Importância por Permutação")
    axes[0].set_xlabel("Queda de F1-Macro")

    for i, (nome, (tr_l, vl_l)) in enumerate(historicos.items()):
        axes[1].plot(tr_l, color=PALETTE[i % len(PALETTE)], label=nome, alpha=0.8)
    axes[1].set_title("Curvas de Perda (Treino) por Conjunto de Atributos")
    axes[1].set_xlabel("Época"); axes[1].set_ylabel("Loss"); axes[1].legend(fontsize=7)

    plt.tight_layout()
    salvar_fig(fig, "q5_atributos.png")

    # Melhor subconjunto
    best_attr = df.loc[df["F1 Val."].idxmax(), "Conjunto de Atributos"]
    best_cols  = experimentos[best_attr]
    print(f"\n  Melhor subconjunto de atributos: '{best_attr}' (cols={len(best_cols)})")

    return best_cols, feature_names


# Q6 – VALIDAÇÃO INICIAL: CONJUNTOS DE TREINAMENTO E TESTE
def questao_6(X_main, y_main, X_test, y_test, top4_df, lr, mom, best_cols):
    """
    Avalia as 4 melhores topologias da Q3 no conjunto de teste.
    """
    print("\n" + "="*70)
    print("Q6 – VALIDAÇÃO INICIAL: TREINAMENTO E TESTE")
    print("="*70)

    scaler = StandardScaler()
    X_main_s = scaler.fit_transform(X_main[:, best_cols])
    X_test_s  = scaler.transform(X_test[:, best_cols])

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_main_s, y_main, test_size=0.20, random_state=42, stratify=y_main
    )

    registros = []
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Q6 – Curvas de Validação: 4 Melhores Topologias", fontsize=14, fontweight="bold")

    melhor_f1_test = -1
    melhor_config  = None

    for idx, (ax, (_, row)) in enumerate(zip(axes.flatten(), top4_df.iterrows())):
        topo  = tuple([row["Neurônios/Camada"]] * int(row["Camadas"]))
        label = row["Topologia"]
        print(f"  Topologia {label} ...", end=" ", flush=True)

        mlp = MLPClassifier(
            hidden_layer_sizes=topo, activation="relu",
            solver="sgd", learning_rate_init=lr, momentum=mom,
            max_iter=1, warm_start=True, random_state=42
        )
        tr_l, vl_l, tr_a, vl_a = [], [], [], []

        for ep in range(150):
            try:
                mlp.fit(X_tr, y_tr)
                tr_l.append(log_loss(y_tr, mlp.predict_proba(X_tr)))
                vl_l.append(log_loss(y_val, mlp.predict_proba(X_val)))
                tr_a.append(accuracy_score(y_tr, mlp.predict(X_tr)))
                vl_a.append(accuracy_score(y_val, mlp.predict(X_val)))
            except Exception:
                break

        acc_test = accuracy_score(y_test, mlp.predict(X_test_s))
        f1_test  = f1_score(y_test, mlp.predict(X_test_s), average="macro", zero_division=0)
        f1_val   = f1_score(y_val,  mlp.predict(X_val),    average="macro", zero_division=0)
        acc_tr   = tr_a[-1] if tr_a else 0
        loss_tr  = tr_l[-1] if tr_l else 9999
        loss_vl  = vl_l[-1] if vl_l else 9999

        if f1_test > melhor_f1_test:
            melhor_f1_test = f1_test
            melhor_config  = {"topo": topo, "label": label, "mlp": mlp}

        registros.append({
            "Topologia": label,
            "Loss Treino": round(loss_tr, 4),
            "Loss Val.":   round(loss_vl, 4),
            "Acc Treino":  round(acc_tr, 4),
            "Acc Val.":    round(f1_val, 4),
            "Acc Teste":   round(acc_test, 4),
            "F1 Teste":    round(f1_test, 4),
        })
        print(f"f1_test={f1_test:.4f}  acc_test={acc_test:.4f}")

        ax2 = ax.twinx()
        ax.plot(tr_l, color=PALETTE[0], alpha=0.8, label="Loss Treino")
        ax.plot(vl_l, color=PALETTE[1], ls="--", alpha=0.8, label="Loss Val.")
        ax2.plot(tr_a, color=PALETTE[2], alpha=0.5, label="Acc Treino")
        ax2.plot(vl_a, color=PALETTE[3], ls="--", alpha=0.5, label="Acc Val.")
        ax.set_title(f"Topologia {label}  |  F1-Teste={f1_test:.3f}")
        ax.set_xlabel("Época"); ax.set_ylabel("Loss"); ax2.set_ylabel("Acurácia")
        lines1, lbl1 = ax.get_legend_handles_labels()
        lines2, lbl2 = ax2.get_legend_handles_labels()
        ax.legend(lines1+lines2, lbl1+lbl2, fontsize=7)

    plt.tight_layout()
    salvar_fig(fig, "q6_curvas_validacao.png")

    df = pd.DataFrame(registros)
    print("\n  Resultados Q6:")
    print(df.to_string(index=False))
    tabela_resultados(df, "q6_resultados.csv")

    # ── Matriz de Confusão (melhor rede) ──────────────────────────────────
    y_pred_best = melhor_config["mlp"].predict(X_test_s)
    cm = confusion_matrix(y_test, y_pred_best)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=range(7), yticklabels=range(7))
    ax.set_title(f"Q6 – Matriz de Confusão: {melhor_config['label']}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predito"); ax.set_ylabel("Real")
    plt.tight_layout()
    salvar_fig(fig, "q6_confusion_matrix.png")

    print(f"\n  Melhor configuração: {melhor_config['label']}  F1-Teste={melhor_f1_test:.4f}")
    return melhor_config["topo"], scaler, best_cols


# Q7 – VALIDAÇÃO CRUZADA k-FOLD
def questao_7(X, y, best_topo, lr, mom, best_cols):
    """
    Validação cruzada k=10 (estratificado) da melhor configuração.
    """
    print("\n" + "="*70)
    print("Q7 – VALIDAÇÃO CRUZADA K-FOLD (k=10)")
    print("="*70)

    K = 10
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

    X_sub = X[:, best_cols]

    registros = []
    curvas_fold = []

    for fold, (tr_idx, ts_idx) in enumerate(skf.split(X_sub, y), 1):
        X_tr, X_ts = X_sub[tr_idx], X_sub[ts_idx]
        y_tr, y_ts = y[tr_idx],     y[ts_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_ts_s = scaler.transform(X_ts)

        print(f"  Fold {fold:2d}/{K} ...", end=" ", flush=True)
        t0 = time.time()

        mlp = MLPClassifier(
            hidden_layer_sizes=best_topo, activation="relu",
            solver="sgd", learning_rate_init=lr, momentum=mom,
            max_iter=1, warm_start=True, random_state=42
        )
        loss_curve = []
        for ep in range(150):
            try:
                mlp.fit(X_tr_s, y_tr)
                loss_curve.append(log_loss(y_ts, mlp.predict_proba(X_ts_s)))
            except Exception:
                break

        elapsed  = time.time() - t0
        acc_test = accuracy_score(y_ts, mlp.predict(X_ts_s))
        f1_test  = f1_score(y_ts, mlp.predict(X_ts_s), average="macro", zero_division=0)
        loss_test= log_loss(y_ts, mlp.predict_proba(X_ts_s))

        curvas_fold.append(loss_curve)
        registros.append({
            "Fold": fold,
            "Loss Teste": round(loss_test, 4),
            "Acurácia":   round(acc_test, 4),
            "F1-Macro":   round(f1_test, 4),
            "Tempo (s)":  round(elapsed, 2)
        })
        print(f"loss={loss_test:.4f}  acc={acc_test:.4f}  f1={f1_test:.4f}")

    df = pd.DataFrame(registros)

    # Estatísticas 
    stats = df[["Loss Teste","Acurácia","F1-Macro"]].agg(["mean","std"])
    print("\n  Resultados Q7 (por fold):")
    print(df.to_string(index=False))
    print("\n  Estatísticas consolidadas:")
    print(stats.round(4).to_string())

    tabela_resultados(df, "q7_resultados_folds.csv")
    tabela_resultados(stats.reset_index().rename(columns={"index":"Estatística"}),
                      "q7_estatisticas.csv")

    # Gráficos 
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig)
    fig.suptitle("Q7 – Validação Cruzada 10-Fold", fontsize=15, fontweight="bold")

    # Curvas de loss
    ax1 = fig.add_subplot(gs[0, :])
    for i, (curve, fold) in enumerate(zip(curvas_fold, range(1, K+1))):
        ax1.plot(curve, alpha=0.6, label=f"Fold {fold}", color=plt.cm.tab10(i/K))
    ax1.set_title("Curva de Loss (Teste) por Fold")
    ax1.set_xlabel("Época"); ax1.set_ylabel("Loss"); ax1.legend(ncol=5, fontsize=8)

    # Barras F1
    ax2 = fig.add_subplot(gs[1, 0])
    bars = ax2.bar(df["Fold"], df["F1-Macro"], color=PALETTE[2], edgecolor="white")
    ax2.axhline(df["F1-Macro"].mean(), color="red", ls="--", lw=1.5, label=f"Média={df['F1-Macro'].mean():.3f}")
    ax2.set_title("F1-Macro por Fold"); ax2.set_xlabel("Fold"); ax2.set_ylabel("F1-Macro")
    ax2.legend(); ax2.set_ylim(0, 1)

    # Boxplot
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.boxplot(
        [df["Loss Teste"], df["Acurácia"], df["F1-Macro"]],
        labels=["Loss Teste","Acurácia","F1-Macro"],
        patch_artist=True,
        boxprops=dict(facecolor=PALETTE[0], color="white"),
        medianprops=dict(color="white", lw=2)
    )
    ax3.set_title("Distribuição das Métricas entre Folds")

    plt.tight_layout()
    salvar_fig(fig, "q7_validacao_cruzada.png")

    print(f"\n  Acurácia média  : {df['Acurácia'].mean():.4f} ± {df['Acurácia'].std():.4f}")
    print(f"  F1-Macro médio  : {df['F1-Macro'].mean():.4f} ± {df['F1-Macro'].std():.4f}")
    print(f"  Loss médio      : {df['Loss Teste'].mean():.4f} ± {df['Loss Teste'].std():.4f}")

    if df["F1-Macro"].std() < 0.02:
        print("  Conclusão: modelo CONSISTENTE e robusto entre partições.")
    else:
        print("  Conclusão: variância moderada entre partições — possível melhoria com regularização.")


# RELATÓRIO RESUMO FINAL
def gerar_resumo(topology, lr, mom, best_topo, best_cols, feature_names):
    linhas = [
        "=" * 65,
        "RELATÓRIO FINAL – ESTUDO DIRIGIDO MLP BACKPROPAGATION",
        "Base de Dados: Dry Bean Dataset (UCI) | 7 classes",
        "=" * 65,
        "",
        "Q1 – EXPLORAÇÃO INICIAL",
        f"  Topologia base     : {topology}",
        f"  Ativação ocultas   : ReLU",
        f"  Ativação saída     : Softmax (implícita no MLPClassifier)",
        f"  Otimizador         : Adam  |  LR={0.001}  |  Épocas=100",
        f"  Perda              : Categorical Cross-Entropy",
        "",
        "Q2 – HIPERPARÂMETROS",
        f"  Melhor LR          : {lr}",
        f"  Melhor Momentum    : {mom}",
        f"  Grid testado       : LR ∈ [0.001,0.01,0.1,0.5] × Mom ∈ [0.5,0.7,0.9]",
        "",
        "Q3 – TOPOLOGIA",
        f"  Melhor topologia   : {best_topo}",
        f"  Testado            : 1–3 camadas × 10–100 neurônios",
        "",
        "Q4 – DADOS",
        f"  Frações testadas   : 20%,40%,60%,80%,100%",
        f"  Divisão            : Estratificada (20% teste fixo)",
        "",
        "Q5 – ATRIBUTOS",
        f"  Melhor subconjunto : {len(best_cols)} atributos",
        f"  Estratégias        : Todos / Top permutação / Sem alta corr. / Aleatório",
        "",
        "Q6 – VALIDAÇÃO INICIAL",
        f"  4 topologias avaliadas no conjunto de teste",
        f"  Veja q6_resultados.csv e q6_confusion_matrix.png",
        "",
        "Q7 – VALIDAÇÃO CRUZADA",
        f"  k = 10 folds estratificados",
        f"  Veja q7_resultados_folds.csv e q7_estatisticas.csv",
        "",
        f"Arquivos gerados em: {OUT_DIR.resolve()}",
        "=" * 65,
    ]
    txt = "\n".join(linhas)
    print("\n" + txt)
    (OUT_DIR / "relatorio_resumo.txt").write_text(txt, encoding="utf-8")


# MAIN
def main():
    print("\n" + "="*70)
    print("  ESTUDO DIRIGIDO – MLP BACKPROPAGATION | DRY BEAN DATASET")
    print("="*70 + "\n")

    # Carregamento e pré-processamento 
    X_raw, y, classes = carregar_dados()
    feature_names = [f"feat_{i}" for i in range(X_raw.shape[1])]

    # Normalização Z-score global 
    scaler_global = StandardScaler()
    X = scaler_global.fit_transform(X_raw)

    # Divisão principal: 80% experimentos | 20% teste final (fixo)
    X_main, X_test, y_main, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_tr70, X_val, y_tr70, y_val = train_test_split(
        X_main, y_main, test_size=0.25, random_state=42, stratify=y_main
    )  # 60/20/20 do total

    print(f"\n  Divisão dos dados:")
    print(f"    Treino (Q1/Q2): {len(X_tr70)} exemplos")
    print(f"    Validação     : {len(X_val)} exemplos")
    print(f"    Teste (fixo)  : {len(X_test)} exemplos")

    # Execução das questões
    topology, lr_q1, solver = questao_1(X_tr70, y_tr70)
    best_lr, best_mom       = questao_2(X_tr70, y_tr70, topology)
    best_topo, top4_df      = questao_3(X_tr70, y_tr70, X_val, y_val, best_lr, best_mom)
    _                       = questao_4(X, y, best_topo, best_lr, best_mom)
    best_cols, feat_names   = questao_5(X_main, y_main, X_test, y_test,
                                        feature_names, best_topo, best_lr, best_mom)
    final_topo, _, _        = questao_6(X_main, y_main, X_test, y_test,
                                        top4_df, best_lr, best_mom, best_cols)
    questao_7(X, y, final_topo, best_lr, best_mom, best_cols)

    gerar_resumo(topology, best_lr, best_mom, final_topo, best_cols, feature_names)

    print(f"\n  Todos os resultados foram salvos em: {OUT_DIR.resolve()}")
    print("  Arquivos gerados:")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"    {f.name}")


if __name__ == "__main__":
    main()