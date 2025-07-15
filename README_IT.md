# YOLOv10/v12 Training & Analysis Suite

**Autore: Valerio Marchetti**

Suite completa per l'addestramento e l'analisi di modelli YOLOv10/v12 con supporto per ablation studies, tracking delle emissioni CO2, integrazione TensorBoard e strumenti di visualizzazione.

## Caratteristiche

- ✅ Addestramento YOLOv10/YOLOv12 con Ultralytics
- ✅ Ablation studies sistematici per ottimizzazione modelli
- ✅ Tracking emissioni CO2 con CodeCarbon
- ✅ Integrazione TensorBoard per monitoraggio in tempo reale
- ✅ Integrazione Wandb per gestione esperimenti
- ✅ Supporto pesi preaddestrati
- ✅ Test automatico del modello
- ✅ Logging dettagliato
- ✅ Gestione memoria GPU
- ✅ Resume addestramento da checkpoint
- ✅ Configurazioni YAML flessibili
- ✅ Scripts avanzati per analisi e visualizzazione metriche
- ✅ Supporto Docker per deployment
- ✅ Esportazione automatica delle metriche

## Installazione

### 1. Clona/Scarica i file necessari

Assicurati di avere i seguenti file:
- `train_yolo.py` - Script principale di addestramento con ablation studies
- `requirements.txt` - Dipendenze Python
- `export_metrics.py` - Script per esportazione metriche
- `plot_curves.py` - Script per plotting delle curve di training
- `plot_grouped_metrics.py` - Script per visualizzazione metriche raggruppate
- `visualize_metrics.py` - Script per visualizzazioni avanzate
- `Dockerfile` - Configurazione Docker per deployment
- `configs/` - Cartella con configurazioni dei modelli (YAML)
- `dataset_configs/` - Cartella con configurazioni dei dataset (YAML)
- `weights/` - Cartella con pesi preaddestrati

### 2. Setup con Docker (Raccomandato)

Il progetto è configurato per l'uso con Docker. Vedi la sezione [Deployment Docker](#deployment-docker) per istruzioni complete.

**Quick Start:**
```bash
# Build dell'immagine
docker build -t cv2025cuda122img:v1 .

# Crea e avvia container
docker run -p6006:6006 -dit --shm-size=32g --gpus '"device=0"' \
    --name cv2025cuda122containerV1GPU0 \
    -v /path/to/your/workspace:/workspace \
    -v /path/to/your/datasets:/datasets \
    cv2025cuda122img:v1
```

### 3. Setup Locale (Alternativo)

Se preferisci installare localmente senza Docker:

```bash
pip install -r requirements.txt
```

### 4. Verifica l'installazione

```bash
# Con Docker
docker exec -it cv2025cuda122containerV1GPU0 python train_yolo.py --help

# Locale
python train_yolo.py --help
```

## Struttura delle Directory

```
project/
├── train_yolo.py                 # Script principale addestramento con ablation
├── requirements.txt                  # Dipendenze
├── export_metrics.py                # Esportazione metriche
├── plot_curves.py                   # Plotting curve di training
├── plot_grouped_metrics.py          # Visualizzazione metriche raggruppate
├── visualize_metrics.py             # Visualizzazioni avanzate
├── Dockerfile                       # Configurazione Docker
├── configs/                         # Configurazioni modelli
│   ├── YOLOv10sLight.yaml          # YOLOv10 Light
│   ├── YOLOv12sLight.yaml          # YOLOv12 Light
│   ├── YOLOv12sLight050.yaml       # YOLOv12 Light 0.5
│   ├── YOLOv12sLightP3P4.yaml      # YOLOv12 Light P3P4
│   ├── YOLOv12sLightP3P4050.yaml   # YOLOv12 Light P3P4 0.5
│   ├── YOLOv12sNormal.yaml         # YOLOv12 Normal
│   ├── YOLOv12sTurbo.yaml          # YOLOv12 Turbo
│   └── YOLOv12sUltraLight.yaml     # YOLOv12 Ultra Light
├── dataset_configs/                 # Configurazioni dataset
│   ├── d3_gray_dataset.yaml        # Dataset D3 Gray
│   └── DVM_1_Grey_dataset.yaml     # Dataset DVM-1 Grey
├── weights/                         # Pesi preaddestrati
│   ├── YOLOv10sLight_d3_grey_cuda22_best.pt
│   ├── YOLOv10sLightTest2_d3_gray_dataset_best.pt
│   ├── YOLOv12sLight_d3_gray_dataset_cuda18_best.pt
│   ├── YOLOv12sLight_d3_grey_cuda22_best.pt
│   ├── YOLOv12sLight_DVM_1_Grey_dataset_best.pt
│   ├── YOLOv12sLight050_d3_gray_dataset_cuda22.pt
│   ├── YOLOv12sLightP3P4_d3_gray_dataset.pt
│   ├── YOLOv12sLightP3P4050_DVM_1_Grey_dataset_best.pt
│   ├── YOLOv12sNormal_d3_gray_dataset_cuda22.pt
│   ├── YOLOv12sTurbo_d3_gray_dataset_cuda18_best.pt
│   └── YOLOv12sUltraLight_d3_gray_dataset_test.pt
└── results/                        # Risultati addestramento (generata)
```

## Utilizzo

### Addestramento Base

```bash
python train_yolo.py --config configs/YOLOv12sTurbo.yaml --dataset dataset_configs/DVM_1_Grey_dataset.yaml
```

### Con Pesi Preaddestrati

```bash
python train_yolo.py --config configs/YOLOv12sTurbo.yaml --dataset dataset_configs/DVM_1_Grey_dataset.yaml --weights weights/yolov12s.pt
```

### Ablation Studies

Esegui ablation studies su diverse configurazioni:

```bash
# Test YOLOv12 Light vs Normal
python train_yolo.py --config configs/YOLOv12sLight.yaml --dataset dataset_configs/d3_gray_dataset.yaml
python train_yolo.py --config configs/YOLOv12sNormal.yaml --dataset dataset_configs/d3_gray_dataset.yaml

# Confronto architetture P3P4
python train_yolo.py --config configs/YOLOv12sLightP3P4.yaml --dataset dataset_configs/DVM_1_Grey_dataset.yaml
python train_yolo.py --config configs/YOLOv12sLightP3P4050.yaml --dataset dataset_configs/DVM_1_Grey_dataset.yaml
```

### Analisi e Visualizzazione

Dopo l'addestramento, usa gli script di analisi:

```bash
# Esporta metriche
python export_metrics.py --run_dir results/

# Crea grafici delle curve di training
python plot_curves.py --results_dir results/

# Visualizza metriche raggruppate per confronti
python plot_grouped_metrics.py --experiments_dir results/

# Visualizzazioni avanzate
python visualize_metrics.py --data_path results/
```

## Parametri Principali

| Parametro | Descrizione | Richiesto |
|-----------|-------------|-----------|
| `--config` | File YAML configurazione modello | ✅ |
| `--dataset` | File YAML configurazione dataset | ✅ |
| `--weights` | Pesi preaddestrati (opzionale) | ❌ |

## Configurazioni Modello Disponibili

Il progetto include diverse configurazioni per ablation studies:

### YOLOv10
- **YOLOv10sLight**: Versione leggera di YOLOv10

### YOLOv12 Base
- **YOLOv12sLight**: Versione leggera standard
- **YOLOv12sNormal**: Versione normale
- **YOLOv12sTurbo**: Versione ottimizzata per velocità
- **YOLOv12sUltraLight**: Versione ultra leggera

### YOLOv12 Varianti Sperimentali
- **YOLOv12sLight050**: Light con scaling 0.5
- **YOLOv12sLightP3P4**: Light con head P3P4
- **YOLOv12sLightP3P4050**: Light P3P4 con scaling 0.5
- **YOLOv12sSM**: Small Model variant
- **YOLOv12sULT**: Ultra variant

### Dataset Supportati
- **d3_gray_dataset**: Dataset D3 in scala di grigi
- **DVM_1_Grey_dataset**: Dataset DVM-1 in scala di grigi

## Pesi Preaddestrati

La cartella `weights/` contiene pesi preaddestrati per accelerare l'addestramento:

### Utilizzo dei Pesi
- **Transfer Learning**: Carica pesi da modelli preaddestrati per fine-tuning
- **Resume Training**: Riprendi addestramento da checkpoint salvati
- **Baseline Models**: Usa modelli già addestrati come baseline per confronti

### Esempi di Utilizzo
```bash
# Con pesi YOLOv10
python train_yolo.py --config configs/YOLOv10sLight.yaml --dataset dataset_configs/d3_gray_dataset.yaml --weights weights/yolov10s.pt

# Con pesi YOLOv12
python train_yolo.py --config configs/YOLOv12sLight.yaml --dataset dataset_configs/DVM_1_Grey_dataset.yaml --weights weights/yolov12s.pt

# Con checkpoint personalizzato
python train_yolo.py --config configs/YOLOv12sTurbo.yaml --dataset dataset_configs/d3_gray_dataset.yaml --weights weights/best.pt
```

## Configurazione Modello (YAML)

Esempio di configurazione YOLOv12:

```yaml
# YOLOv12 Configuration
nc: 1  # Number of classes
scales:
  s: [0.50, 0.50, 1024]  # small

# Backbone architecture
backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  # ... altre layers

# Head architecture  
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  # ... altre layers

# Training parameters
training:
  task: detect
  optimizer: SGD
  lr0: 0.01
  lrf: 0.0001
  momentum: 0.9
  weight_decay: 0.0005
  warmup_epochs: 3
  pretrained: False
  cache: False
  save_dir: "/workspace/results"
  project: "/workspace/results"
  device: 0
  epochs: 200
  batch: 64
  save: True
  verbose: True
  save_period: 10
```

## Configurazione Dataset (YAML)

Esempio di configurazione dataset:

```yaml
# Dataset path configuration
path: /path/to/dataset
train: images/train
val: images/val  
test: images/test

# Class configuration
nc: 1  # Number of classes
names: ['ship']  # Class names
```

## Monitoraggio e Analisi

### TensorBoard
- Monitoraggio in tempo reale delle metriche
- Visualizzazione loss curves, learning rate, mAP
- Accesso via: `tensorboard --logdir=runs/tensorboard --host=0.0.0.0 --port=6006`

### Wandb Dashboard
- Gestione esperimenti avanzata
- Confronto tra diverse configurazioni
- Tracking hyperparameters e metriche

### Scripts di Analisi

1. **export_metrics.py**: Esporta metriche da TensorBoard/Wandb in formato CSV/JSON
2. **plot_curves.py**: Genera grafici delle curve di training e validazione
3. **plot_grouped_metrics.py**: Crea visualizzazioni comparative tra esperimenti
4. **visualize_metrics.py**: Produce visualizzazioni avanzate e report

### Emissioni CO2
- Tracking automatico con CodeCarbon
- Salvate in `results/*/co2_production.txt`
- Dati dettagliati in `results/*/emissions.csv`

## Risultati

I risultati dell'addestramento vengono salvati in:

```
results/
└── ModelName_DatasetName/
    ├── weights/
    │   ├── best.pt              # Migliori pesi
    │   └── last.pt              # Ultimi pesi
    ├── args.yaml                # Argomenti addestramento
    ├── co2_production.txt       # Emissioni CO2
    ├── emissions.csv            # Dettagli emissioni CodeCarbon
    ├── labels.jpg               # Visualizzazione labels
    ├── train_batch*.jpg         # Batch di training
    ├── runs/
    │   └── detect/
    │       └── ModelName_DatasetName_test/
    │           ├── confusion_matrix.png
    │           ├── results.png
    │           └── predictions/
    └── tensorboard/             # Log TensorBoard
        └── events.out.tfevents.*
```

## Esempi di Ablation Studies

### 1. Confronto Architetture Base

```bash
# YOLOv10 vs YOLOv12
python train_yolo.py --config configs/YOLOv10sLight.yaml --dataset dataset_configs/d3_gray_dataset.yaml
python train_yolo.py --config configs/YOLOv12sLight.yaml --dataset dataset_configs/d3_gray_dataset.yaml
```

### 2. Studio Scaling Factors

```bash
# Confronto scaling factors
python train_yolo.py --config configs/YOLOv12sLight.yaml --dataset dataset_configs/DVM_1_Grey_dataset.yaml
python train_yolo.py --config configs/YOLOv12sLight050.yaml --dataset dataset_configs/DVM_1_Grey_dataset.yaml
```

### 3. Valutazione Head Architecture

```bash
# Standard vs P3P4 head
python train_yolo.py --config configs/YOLOv12sLight.yaml --dataset dataset_configs/d3_gray_dataset.yaml
python train_yolo.py --config configs/YOLOv12sLightP3P4.yaml --dataset dataset_configs/d3_gray_dataset.yaml
python train_yolo.py --config configs/YOLOv12sLightP3P4AA.yaml --dataset dataset_configs/d3_gray_dataset.yaml
```

### 4. Fine-tuning con Transfer Learning

```bash
# Fine-tuning da pesi preaddestrati
python train_yolo.py \
    --config configs/YOLOv12sTurbo.yaml \
    --dataset dataset_configs/DVM_1_Grey_dataset.yaml \
    --weights weights/yolov12s.pt
```

## Analisi Post-Training

Dopo aver completato gli esperimenti, usa gli script di analisi:

```bash
# 1. Esporta tutte le metriche
python export_metrics.py --experiments_dir results/

# 2. Genera confronti visuali
python plot_grouped_metrics.py --experiments results/YOLOv12sLight_* results/YOLOv12sNormal_*

# 3. Crea report completo
python visualize_metrics.py --results_dir results/ --output_dir analysis/
```

## Deployment Docker

### Setup Ambiente UnivPM

Il progetto è configurato per l'utilizzo con il server GPU di UnivPM. Segui questa procedura:

#### 1. Build dell'Immagine Docker

```bash
docker build -t cv2025cuda122img:v1 .
```

#### 2. Creazione Container

**Configura i percorsi:**
- `/path/to/your/workspace`: Directory di lavoro contenente il codice e dove saranno salvati i risultati
- `/path/to/your/datasets`: Directory contenente i dataset per l'addestramento

Per utilizzare GPU diverse (GPU0 o GPU2) del server UnivPM:

**Container per GPU0:**
```bash
docker run -p6006:6006 -dit --shm-size=32g --gpus '"device=0"' \
    --name cv2025cuda122containerV1GPU0 \
    -v /path/to/your/workspace:/workspace \
    -v /path/to/your/datasets:/datasets \
    cv2025cuda122img:v1
```

**Container per GPU2:**
```bash
docker run -p6007:6006 -dit --shm-size=32g --gpus '"device=2"' \
    --name cv2025cuda122containerV1GPU2 \
    -v /path/to/your/workspace:/workspace \
    -v /path/to/your/datasets:/datasets \
    cv2025cuda122img:v1
```

#### 3. Accesso al Container

```bash
# Per GPU0
docker exec -it cv2025cuda122containerV1GPU0 bash

# Per GPU2
docker exec -it cv2025cuda122containerV1GPU2 bash
```

#### 4. Gestione Sessioni con Tmux

Una volta dentro il container, usa `tmux` per gestire sessioni persistenti:

**Creare nuova sessione:**
```bash
tmux new -s yolo_train2
```

**Uscire dalla sessione (lasciandola attiva):**
```
CTRL+B + D
```

**Riprendere sessione esistente:**
```bash
tmux attach -t yolo_train2
```

**Lista sessioni attive:**
```bash
tmux ls
```

#### 5. Esecuzione Training

All'interno della sessione tmux:

```bash
# Esempio di training
python train_yolo.py \
    --config configs/YOLOv12sTurbo.yaml \
    --dataset dataset_configs/DVM_1_Grey_dataset.yaml \
    --weights weights/yolov12s.pt
```

#### 6. Accesso TensorBoard

TensorBoard è accessibile via browser:
- **GPU0**: `http://server-ip:6006`
- **GPU2**: `http://server-ip:6007`

#### Comandi Utili Tmux
```bash
# Lista tutte le sessioni
tmux ls

# Crea sessione con nome specifico
tmux new -s ablation_study_1

# Termina sessione
tmux kill-session -t yolo_train2

# Rinomina sessione corrente
tmux rename-session new_name
```

#### Monitoraggio GPU
```bash
# Verifica utilizzo GPU
nvidia-smi

# Monitoraggio continuo
watch -n 1 nvidia-smi
```

### Workflow Completo UnivPM

Esempio di workflow completo per ablation study:

```bash
# 1. Build immagine (una volta sola)
docker build -t cv2025cuda122img:v1 .

# 2. Crea container GPU0
docker run -p6006:6006 -dit --shm-size=32g --gpus '"device=0"' \
    --name cv2025cuda122containerV1GPU0 \
    -v /path/to/your/workspace:/workspace \
    -v /path/to/your/datasets:/datasets \
    cv2025cuda122img:v1

# 3. Accedi al container
docker exec -it cv2025cuda122containerV1GPU0 bash

# 4. Crea sessione tmux
tmux new -s ablation_yolov12

# 5. Avvia training
python train_yolo.py \
    --config configs/YOLOv12sLight.yaml \
    --dataset dataset_configs/DVM_1_Grey_dataset.yaml

# 6. Esci da tmux (CTRL+B + D) per lasciare il training in background

# 7. In un'altra sessione, monitora TensorBoard via browser:
# http://server-ip:6006

# 8. Per riprendere il monitoraggio:
tmux attach -t ablation_yolov12
```

## Note sulla Versione

- **Versione attuale**: v7 (train_yolo.py)
- **Caratteristiche principali**: Ablation studies, TensorBoard integration, CO2 tracking
- **Compatibilità**: YOLOv10/v12, Python 3.8+, CUDA 12.2+

## Licenza

Questo progetto segue le licenze delle librerie utilizzate (Ultralytics YOLO, etc.).

## Autore

**Valerio Marchetti**  
Progetto sviluppato per il corso di Computer Vision and Deep Learning, anno 2025

---

**Creato per il progetto CVDL - Computer Vision and Deep Learning**
