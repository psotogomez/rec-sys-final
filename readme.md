# Recommender System Final Project

Este repositorio contiene diferentes modelos de sistemas de recomendación. Cada carpeta corresponde a un modelo específico. A continuación se describen brevemente los modelos incluidos:

## Modelos

### 1. GRU4REC
Ubicación: `./gru4rec/`

Para ejecutar este modelo se debe utilizar en jupyter. Dependiendo el dataset a utilizar puede ejecutar yoochoose en GRU4Rec_official_pytorch_implementation.ipynb o RetailRocket en GRU4Rec_official_pytorch_retail_rocket.ipynb

### 2. LightGCN
Ubicación: `./lightgcn/`

para ejecutar este modelo se debe correr light.py donde dentro del archivo se puede elegir que dataset será utlizado

### 3. MULTVAE
Ubicación: `./MULTVAE/`

para ejecutar este modelo se debe correr run_VAE.py, dentro del arhivo se puede modificar el dataset a utilizar.


## Cómo Instalar dependencias

1. Clona el repositorio:
    ```sh
    git clone https://github.com/psotogomez/rec-sys-final.git && cd rec-sys-final
    ```

2.Inicializa el ambiente de conda:
    ```sh
    conda env create -f spec-file.yml
    ```