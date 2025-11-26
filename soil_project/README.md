# Soil Type Classification from Telemetry

Projeto para construir um dataset supervisionado de classificacao de solo a partir de telemetria off-road (IMU, velocidade etc.). Nesta versao inicial, o rotulo de solo vem apenas de heuristicas terramecanicas aplicadas a clusters de janelas de telemetria.

## Estrutura do projeto

```
soil_project/
├─ data/                        # Dados do projeto
│  ├─ raw/                      # Dados brutos (arquivos originais do sensor)
│  │  └─ SensorData/            # Sessoes de coleta (uma pasta por gravacao)
│  └─ processed/                # Dados processados/unificados
│     └─ telemetry.csv          # Telemetria unificada (gerada pelo merge_telemetry.py)
├─ models/                      # Modelos treinados (ex.: soil_classifier.pkl)
├─ notebooks/                   # Notebooks opcionais para EDA/experimentos
├─ scripts/                     # Scripts de alto nivel da pipeline
│  ├─ merge_telemetry.py        # Une os CSVs brutos em um unico telemetry.csv
│  ├─ cluster_elbow.py          # Calcula e plota grafico de elbow (escolha de K)
│  ├─ build_dataset.py          # Constroi o dataset supervisionado final
│  ├─ train_model.py            # Treina e avalia o classificador
│  └─ run_full_pipeline.py      # Executa build + treino em sequencia
├─ soil_dataset/                # Pacote Python com a logica de dados/modelo
│  ├─ __init__.py
│  ├─ config.py                 # Caminhos, seeds, parametros (WINDOW_SIZE, etc.)
│  ├─ data_loading.py           # Funcoes de carregamento de CSV
│  ├─ soil_processing.py        # Reservado para futura integracao com dataset de solo
│  ├─ telemetry_processing.py   # Janelamento e features de telemetria
│  ├─ clustering.py             # Clusterizacao das janelas de telemetria
│  ├─ cluster_to_soil_mapping.py# Heuristicas terramecanicas cluster -> tipo de solo / modo
│  ├─ fusion.py                 # Reservado para futura fusao com embeddings de solo
│  └─ training.py               # Treino e avaliacao do classificador
├─ tests/                       # Reservado para testes automatizados
├─ requirements.txt             # Dependencias do projeto
└─ README.md                    # Este arquivo
```

## Geracao do arquivo `telemetry.csv` (merge_telemetry.py)

O script `scripts/merge_telemetry.py` une os dados brutos de cada sessao em um unico arquivo `data/processed/telemetry.csv`.

Estrutura esperada dos dados brutos:

```
soil_project/data/raw/SensorData/
├─ 2020-07-28-06-01-11/
│  ├─ accelerometer_calibrated_split.csv
│  ├─ gyroscope_calibrated_split.csv
│  ├─ magnetometer_split.csv       # (nao usado por enquanto)
│  ├─ gps.csv                      # (nao usado por enquanto)
│  └─ record.csv
├─ 2020-09-23-16-10-10/
│  └─ ...
└─ ...
```

Para cada pasta de sessao, o script:

1) Le acelerometro/giroscopio (arquivos calibrados) e record.csv.
2) Usa `utc_s (s)` e `utc_ms (ms)` para alinhar sinais:
   - Une acelerometro + giroscopio (mesma frequencia) por chave exata (`utc_s (s)`, `utc_ms (ms)`).
   - Cria `time_sec = utc_s (s) + utc_ms (ms)/1000`.
3) No record.csv, cria `time_sec = utc_s (s)` e faz `merge_asof` (tempo mais proximo) para juntar:
   - `distance (m)`, `enhanced_speed (m/s)`, `enhanced_altitude (m)`.
4) Cria `speed = enhanced_speed (m/s)` (coluna simplificada para a pipeline).
5) Adiciona `session` com o nome da pasta (ex.: `2020-07-28-06-01-11`).
6) Concatena todas as sessoes e salva em `data/processed/telemetry.csv`.

## Principais variaveis em `telemetry.csv`

- `utc_s (s)`, `utc_ms (ms)`: carimbo de tempo bruto.
- `time_sec`: tempo continuo em segundos (alinhamento entre sinais).
- `accel_* (counts)`: leituras brutas do acelerometro.
- `calibrated_accel_* (g)` e `(m/s^2)`: aceleracao calibrada, base para vibracao/RMS.
- `gyro_* (counts)` e `calibrated_gyro_* (deg/s)`: velocidades angulares calibradas.
- `distance (m)`: distancia percorrida acumulada.
- `enhanced_speed (m/s)` e `speed`: velocidade do veiculo.
- `enhanced_altitude (m)`: altitude (ajuda a inferir rampa).
- `session`: identificador da sessao.
- Colunas `Unnamed: 0_*`: indices gerados ao salvar/concatenar; podem ser removidos se desejar.

## Pre-requisitos

- Python 3.9+ recomendado.
- Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Como usar (versao inicial, sem embeddings)

1) Gerar o `telemetry.csv` unificado a partir dos CSVs brutos:
```bash
cd soil_project
python scripts/merge_telemetry.py
```

2) Gerar o dataset final supervisionado (janelamento + clusters + rotulo de solo):
```bash
python scripts/build_dataset.py
```
> Antes de fixar o numero de clusters, use `scripts/cluster_elbow.py` para gerar o grafico de elbow e escolher `N_CLUSTERS` em `soil_dataset/config.py`.

3) Treinar o classificador de tipo de solo:
```bash
python scripts/train_model.py
```

4) Executar a pipeline completa (build + treino):
```bash
python scripts/run_full_pipeline.py
```

## Como os modos de operacao foram definidos

- Features usadas no clustering (janelas de telemetria): `mean_speed`, `std_speed`, `mean_accel_z`, `std_accel_z`, `mean_gyro_y`, `std_gyro_y`, `speed_drop`, `rms_accel_z`. Elas refletem ritmo (speed), rugosidade/vibracao (accel/gyro) e tendencia de ganho/perda de velocidade na janela (`speed_drop`).
- Clustering: KMeans com `N_CLUSTERS` em `soil_dataset/config.py`, sem rótulo (nao supervisionado).
- Mapeamento heuristico para **operation_mode** (`soil_dataset/cluster_to_soil_mapping.py`):
  - `mean_speed`: slow (< 3 m/s), cruise (3–4 m/s), fast (>= 4 m/s).
  - `rms_accel_z`: smooth (< 10.8), medium (10.8–12.0), rough (>= 12.0). Valores maiores indicam terreno mais asperto/vibracao.
  - `speed_drop`: descent_brake (<= -0.02), flat (-0.02 a 0.04), climb_push (> 0.04). Diferenca entre velocidade inicial e final da janela, sugerindo rampa ou tracao extra.
  - O modo final e montado como `speedLevel_vibrationLevel_slopeHint`, ex.: `fast_rough_climb_push`, `slow_smooth_flat`.
- Resultado atual com K=6 (dados fornecidos): `{0: cruise_medium_flat, 1: slow_smooth_flat, 2: fast_rough_flat, 3: fast_rough_climb_push, 4: fast_rough_climb_push, 5: cruise_medium_descent_brake}`. A coluna `operation_mode` esta em `data/processed/telemetry_windowed_with_clusters.csv` e `data/processed/final_soil_telemetry_dataset.csv`.
- Solo: o mapeamento ainda cai todo em "gravel" porque faltam sinais de slip/torque (`slip_ratio`, `mean_motor_current`). Para separar solos, adicione esses sinais ou ajuste os thresholds quando eles existirem.

## Observacoes

- As heuristicas de solo sao simplificadas e devem ser refinadas com base em conhecimento fisico/terramecanico e dados reais.
- Integracao futura com dataset de solo (ex.: LUCAS Soil) e uso de embeddings PCA podem ser adicionados sem quebrar a pipeline atual.
