# Vacuum Cleaner Project – Agentes Inteligentes no Mundo do Aspirador

Este projeto implementa e avalia dois tipos de agentes artificiais (reativo simples e baseado em modelo) para o problema do aspirador de pó num ambiente de grade retangular com obstáculos. O ambiente é determinístico e parcialmente observável: o agente só percebe se a célula atual está suja e se há um obstáculo imediatamente à frente. O objetivo é limpar toda a sujeira.

## Estrutura do Código

- `src/environment.py`: Simula o mundo (grade, sujeira, obstáculos, sensores e atuadores).
- `src/simple_agent.py`: Agente reativo simples (regras condição‑ação).
- `src/model_agent.py`: Agente baseado em modelo (mantém mapa interno, explora usando BFS).
- `src/experiment.py`: Executa experimentos, coleta pontuações, gera CSV e gráficos.
- `results/`: Pasta onde são salvos os resultados (CSV e PNG).

## Requisitos

- Python 3.12+
- Bibliotecas: `numpy`, `pandas`, `matplotlib`

## Instalação e Execução

### Com `uv` (recomendado para desenvolvimento)

```bash
# Instalar uv (se ainda não tiver)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# No diretório do projeto
uv sync                  # cria .venv e instala dependências
uv run python src/experiment.py
```

### Com Docker (ambiente isolado)

```bash
# Construir a imagem
docker build -t vacuum-project .

# Executar o experimento (resultados aparecem na pasta ./results)
docker run --rm -v "$(pwd)/results:/workspace/results" vacuum-project
```

Para reconstruir sem cache (forçar cópia atualizada do código):

```bash
docker build --no-cache -t vacuum-project .
```

## Agentes Implementados

### Agente Reativo Simples (`SimpleReactiveAgent`)
- **Mecanismo**: Regras condição‑ação baseadas apenas na percepção atual.
- **Comportamento**: Se célula suja → aspirar; senão se frente livre → avançar; senão → virar à direita.
- **Não mantém memória** → pode repetir áreas e não explora sistematicamente.

### Agente Baseado em Modelo (`ModelBasedAgent`)
- **Mecanismo**: Mantém um mapa interno (coordenadas relativas, estado das células), uma fila de fronteira (células a explorar) e um histórico de visitas. Utiliza BFS para planejar caminhos até células sujas ou desconhecidas.
- **Comportamento**: Aspira sujeira imediatamente; se não há sujeira, planeja rota para a célula suja mais próxima; se nenhuma sujeira é conhecida, explora células‑fronteira; ao final, desliga‑se (`shut_off`).
- **Atualização do modelo**: Após cada ação, o mapa é atualizado com a percepção e o sucesso do movimento.

## Medidas de Avaliação

- **Métrica A**: +1 ponto para cada quadrado limpo (sem penalidade por movimentos).
- **Métrica B**: +1 ponto para cada quadrado limpo **e** –1 ponto por cada movimento de avanço (virar não é penalizado).

## Experimentos Realizados

- **Tamanho do ambiente**: 6×6 células.
- **Probabilidades**: 30% de sujeira nas células livres, 10% de obstáculos.
- **Número de episódios**: 100 (cada episódio com configuração aleatória de sujeira, obstáculos e posição inicial do agente).
- **Limite de passos por episódio**: 500.

## Resultados Obtidos (média ± desvio padrão)

| Agente               | Métrica A          | Métrica B            |
|----------------------|--------------------|----------------------|
| Reativo Simples      | 4,53 ± 2,54        | -343,86 ± 51,30      |
| Baseado em Modelo    | 10,54 ± 2,58       | -64,70 ± 16,97       |

### Interpretação sumária

- O agente baseado em modelo limpa em média 10,5 sujeiras (quase todas, pois o total esperado é ~10,8), enquanto o reativo limpa apenas 4,5.
- Na métrica B, o modelo sofre muito menos penalidades (cerca de -65 contra -344 do reativo), demonstrando maior racionalidade ao planejar caminhos eficientes.

Os gráficos (barplot e boxplot) e os dados completos (CSV) estão disponíveis na pasta `results/`.

## Licença

Projeto acadêmico para a disciplina de Inteligência Computacional.
