# XAI com LIME e Small Language Models

Este projeto explora o uso de **Explainable AI (XAI)** para tornar
decisões de modelos de machine learning mais compreensíveis.

A proposta combina:

-   **LIME (Local Interpretable Model-agnostic Explanations)** para
    identificar quais variáveis influenciam uma predição.
-   **Small Language Models (SLMs)** para traduzir os valores numéricos
    do LIME em explicações em linguagem natural.

O experimento foi realizado utilizando o dataset **Breast Cancer
Wisconsin (Diagnostic)**, com o objetivo de analisar classificações de
tumores como benignos ou malignos.

------------------------------------------------------------------------

# Objetivos do Projeto

-   Treinar um modelo de classificação para o dataset de câncer de mama
-   Gerar explicações locais utilizando LIME
-   Avaliar a **estabilidade das explicações** variando o número de
    perturbações
-   Utilizar um **Small Language Model (Qwen2.5)** para transformar os
    resultados do LIME em explicações textuais
-   Comparar diferentes estilos de prompts para interpretação das
    explicações

------------------------------------------------------------------------

# Pipeline do Projeto

O fluxo do projeto segue as etapas abaixo:

1.  Carregamento e preparação dos dados
2.  Treinamento do modelo de classificação (Naive Bayes)
3.  Geração de explicações locais com LIME
4.  Análise de estabilidade das explicações
5.  Tradução das explicações usando um Small Language Model

------------------------------------------------------------------------

# Estrutura do Projeto

    ├── data/
    │   └── processed/          # dados tratados para uso no modelo
    │
    ├── notebooks/
    │   └── experiment_lime_pertubations.ipynb   # experimento principal
    │
    ├── outputs/                # resultados e arquivos gerados
    │
    ├── src/
    │   ├── loader.py           # carregamento dos dados
    │   ├── model.py            # treinamento do modelo
    │   ├── lime_utils.py       # funções relacionadas ao LIME
    │   ├── prompts.py          # prompts usados para o SLM
    │   └── slm_utils.py        # integração com o modelo de linguagem
    │
    └── .gitignore
    └── README.md
