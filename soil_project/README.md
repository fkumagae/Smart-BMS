# Soil Type Classification from Telemetry

Este projeto constrói um dataset supervisionado para classificar tipos de solo a partir de telemetria off-road (IMU, velocidade, etc.).  
Nesta versão inicial, o rótulo de solo é derivado apenas da telemetria via heurísticas terramecânicas aplicadas a clusters de janelas de sinais.

## Estrutura do projeto

```text
soil_project/
├── data/                        # Dados do projeto
│   ├── raw/                     # Dados brutos (arquivos originais do sensor)
│   │   └── SensorData/          # Sessões de coleta (uma pasta por gravação)
│   └── processed/               # Dados processados/unificados
│       └── telemetry.csv        # Telemetria unificada (gerada pelo merge_telemetry.py)
├── models/                      # Modelos treinados (ex.: soil_classifier.pkl)
├── notebooks/                   # Notebooks opcionais para EDA/experimentos
├── scripts/                     # Scripts de alto nível da pipeline
│   ├── merge_telemetry.py       # Une os CSVs brutos em um único telemetry.csv
│   ├── build_dataset.py         # Constrói o dataset supervisionado final
│   ├── train_model.py           # Treina e avalia o classificador
│   └── run_full_pipeline.py     # Executa build + treino em sequência
├── soil_dataset/                # Pacote Python com a lógica de dados/modelo
│   ├── __init__.py
│   ├── config.py                # Caminhos, seeds, parâmetros (WINDOW_SIZE, etc.)
│   ├── data_loading.py          # Funções de carregamento de CSV
│   ├── soil_processing.py       # Reservado para futura integração com dataset de solo
│   ├── telemetry_processing.py  # Janelamento e features de telemetria
│   ├── clustering.py            # Clusterização das janelas de telemetria
│   ├── cluster_to_soil_mapping.py # Heurísticas terramecânicas cluster → tipo de solo
│   ├── fusion.py                # Reservado para futura fusão com embeddings de solo
│   └── training.py              # Treino e avaliação do classificador
├── tests/                       # Reservado para testes automatizados
├── requirements.txt             # Dependências do projeto
├── README.md                    # Este arquivo
└── .gitignore
```

## Geração do arquivo `telemetry.csv` (merge_telemetry.py)

O script `scripts/merge_telemetry.py` é responsável por unir os dados brutos de cada sessão em um único arquivo `data/processed/telemetry.csv`.

Estrutura esperada dos dados brutos:

```text
soil_project/data/raw/SensorData/
├── 2020-07-28-06-01-11/
│   ├── accelerometer_calibrated_split.csv
│   ├── gyroscope_calibrated_split.csv
│   ├── magnetometer_split.csv        # (não usado por enquanto)
│   ├── gps.csv                       # (não usado por enquanto)
│   └── record.csv
├── 2020-09-23-16-10-10/
│   └── ...
└── ...
```

Para cada pasta de sessão, o script:

1. Lê:
   - `accelerometer_calibrated_split.csv`
   - `gyroscope_calibrated_split.csv`
   - `record.csv`
2. Usa as colunas de tempo `utc_s (s)` e `utc_ms (ms)` para:
   - Unir acelerômetro e giroscópio (mesma frequência) por chave exata (`utc_s (s)`, `utc_ms (ms)`).
   - Criar uma coluna de tempo contínuo `time_sec = utc_s (s) + utc_ms (ms)/1000`.
3. No `record.csv`, cria também `time_sec = utc_s (s)` e faz um `merge_asof` (merge por tempo mais próximo) para juntar:
   - `distance (m)`
   - `enhanced_speed (m/s)`
   - `enhanced_altitude (m)`
4. Cria uma coluna de conveniência:
   - `speed` = `enhanced_speed (m/s)` (usada depois pela pipeline).
5. Adiciona uma coluna `session` com o nome da pasta (ex.: `2020-07-28-06-01-11`).
6. Concatena todas as sessões e salva em `data/processed/telemetry.csv`.

Com isso, você passa a ter um único arquivo de telemetria contínua combinando IMU + velocidade/distância/altitude, pronto para janelamento e clusterização.

## Principais variáveis em `telemetry.csv`

Algumas colunas importantes do `telemetry.csv`:

- `utc_s (s)`, `utc_ms (ms)`  
  Carimbo de tempo bruto (segundos e milissegundos) vindo dos sensores.

- `time_sec`  
  Tempo contínuo em segundos (`utc_s + utc_ms/1000`). Usado para alinhar sinais de frequências diferentes.

- `accel_x (counts)`, `accel_y (counts)`, `accel_z (counts)`  
  Leituras brutas do acelerômetro em contagens (valores discretos do ADC).

- `calibrated_accel_x (g)`, `calibrated_accel_y (g)`, `calibrated_accel_z (g)`  
  Aceleração calibrada em unidades de “g” (gravidade).

- `calibrated_accel_x (m/s^2)`, `calibrated_accel_y (m/s^2)`, `calibrated_accel_z (m/s^2)`  
  Aceleração calibrada em metros por segundo ao quadrado, ideal para calcular vibrações (RMS, desvio padrão, etc.).

- `gyro_x (counts)`, `gyro_y (counts)`, `gyro_z (counts)`  
  Leituras brutas do giroscópio em contagens.

- `calibrated_gyro_x (deg/s)`, `calibrated_gyro_y (deg/s)`, `calibrated_gyro_z (deg/s)`  
  Velocidades angulares calibradas em graus por segundo, relacionadas a pitch/roll/yaw e dinâmica do veículo.

- `distance (m)`  
  Distância percorrida acumulada (derivada do GNSS/dispositivo).

- `enhanced_speed (m/s)`  
  Velocidade do veículo em metros por segundo (proveniente do `record.csv`).

- `speed`  
  Cópia de `enhanced_speed (m/s)` com nome simplificado, usada como coluna padrão de velocidade nas próximas etapas da pipeline.

- `enhanced_altitude (m)`  
  Altitude em metros (proveniente do `record.csv`), útil para analisar efeitos de rampa/inclinação.

- `session`  
  Identificador da sessão de coleta (nome da pasta), útil para separar ou filtrar dados por experimento.

- Colunas `Unnamed: 0_*`  
  Índices gerados ao salvar/concatenar CSVs — podem ser ignorados ou removidos em um pré-processamento posterior.

Essas variáveis são a base para extrair features terramecânicas como vibração (RMS de aceleração), variação de velocidade, possíveis proxies de slip (quando houver roda/torque), etc.

## Pré-requisitos

- Python 3.9+ recomendado
- Instalar dependências:

```bash
pip install -r requirements.txt
```

## Como usar (versão inicial, sem embeddings)

Nesta primeira versão, o rótulo de tipo de solo é gerado apenas a partir da telemetria, usando heurísticas terramecânicas (slip, vibração, esforço, etc.) aplicadas sobre clusters de janelas de telemetria. O uso de embeddings derivados de um dataset de solo (como LUCAS Soil) fica como extensão futura.

1. Gerar o `telemetry.csv` unificado a partir dos CSVs brutos:

```bash
cd soil_project
python scripts/merge_telemetry.py
```

2. Gerar o dataset final supervisionado (janelamento + clusters + rótulo de solo):

```bash
python scripts/build_dataset.py
```

3. Treinar o classificador de tipo de solo:

```bash
python scripts/train_model.py
```

4. Executar a pipeline completa (construção do dataset + treino):

```bash
python scripts/run_full_pipeline.py
```

## Observações

- O código assume colunas de exemplo na telemetria (como as listadas acima), podendo ser adaptado ao seu dataset real.
- As heurísticas de mapeamento cluster → tipo de solo são simplificadas e devem ser refinadas com base em conhecimento físico/terramecânico e dados reais.
- A integração com um dataset de solo (ex.: LUCAS Soil) e o uso de embeddings PCA podem ser adicionados posteriormente, sem quebrar a pipeline atual.

