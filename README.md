# 🌳 EcoGuard-Cerrado: Segmentação de Desmatamento via Sensoriamento Remoto

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Fase%201%20(E1)%3A%20Prova%20de%20Conceito-yellow.svg)]()

> **Projeto de Residência Profissional (TIC 44 - CTE-IA)** > **Objetivo:** Propor e validar uma arquitetura computacional capaz de identificar desmatamento recente no bioma Cerrado (Matopiba), em conformidade com as exigências de auditoria ESG da Lei EUDR, superando as limitações dos índices vegetativos tradicionais.

## 📌 O Desafio Tecnológico
Métodos tradicionais de monitoramento baseados em limites de pixel (como o NDVI) perdem a precisão durante a estação seca do Cerrado. A perda natural de folhagem confunde os algoritmos, gerando falsos positivos em áreas de reserva ambiental ou fazendas consolidadas em entressafra. 

Este repositório documenta a especificação da nossa solução: a substituição de cálculos lineares por **Redes Neurais Convolucionais (U-Net)**, capazes de interpretar o contexto espacial e a geometria das bordas antrópicas.

---

## ⚙️ Arquitetura e Pipeline do Projeto

O projeto está sendo modularizado para garantir isolamento de responsabilidades. O nosso pipeline de desenvolvimento segue o fluxo abaixo:

1. **Módulo de Ingestão (`src/data_ingestion/` - *Em desenvolvimento*):** - Consumo dinâmico de imagens Sentinel-2 (L2A) via STAC API (Catálogo AWS).
   - Rasterização de Shapefiles/GeoJSON para compatibilização da verdade terrestre (MapBiomas).
2. **Módulo de Baseline (`src/baseline/` - *Em desenvolvimento*):**
   - Aplicação de limiares matemáticos (NDVI < 0.25) sobre dados reais para geração de métricas de falha (IoU) e comprovação da necessidade de IA.
3. **Módulo de Deep Learning (`src/model/` - *Setup Estrutural Concluído*):**
   - Estruturação da arquitetura U-Net multicanal (RGB + NIR) preparada para a fase de treinamento (Entrega 2).

---

## 📂 Estrutura do Repositório

Nesta primeira fase (Entrega 1), o repositório foi estruturado para receber o fluxo de dados e os scripts que estão sendo refatorados a partir dos nossos testes exploratórios:

```
ecoguard-cerrado/
├── data/                  # Diretório isolado via .gitignore (Proteção de armazenamento)
│   ├── raw/               # Destino das imagens de satélite brutas (.tif)
│   └── processed/         # Destino dos tensores rasterizados prontos para o modelo
├── docs/                  # Documentação oficial, relatórios E1 e diagramas
├── notebooks/             # Ambiente de testes e exploração de dados (EDA)
├── src/                   # Scripts Python modulares (Pipeline principal)
├── requirements.txt       # Mapeamento estrito de dependências
└── README.md              # Este documento
```

---

## 🚀 Reprodutibilidade e Configuração do Ambiente

Garantir que o ambiente de desenvolvimento seja reproduzível é uma premissa deste projeto. Utilizamos o gerenciador **`uv`** para uma criação de ambiente virtual ultrarrápida e segura.

Siga os passos abaixo para instanciar o projeto localmente:

**1. Clone o repositório:**
```
git clone https://github.com/cristianomendes3/ecoguard-cerrado.git
cd ecoguard-cerrado
```

**2. Crie e ative o ambiente virtual (Recomendado o uso do `uv`):**
```
# Criação do ambiente
py -m uv venv

# Ativação no Windows (PowerShell)
.venv\Scripts\activate
# Ativação no Linux/macOS
source .venv/bin/activate
```


**3. Instale as dependências mapeadas:**
```
py -m uv pip install -r requirements.txt
```

*(Nota: Certifique-se de estar com o ambiente virtual ativado para evitar conflitos de escopo global).*