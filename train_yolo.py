# This script is used to train a YOLO model with specified configurations and datasets.
# It includes functionalities for logging, CO2 emissions tracking, and TensorBoard integration.
# Author: Valerio Marchetti
# Date: 2024-01-15
# version: 7

import os
import yaml
import wandb
from ultralytics import YOLO
from codecarbon import EmissionsTracker
import logging
import sys
import time
import json
import pandas as pd
import torch
from pathlib import Path
import argparse
from torch.utils.tensorboard import SummaryWriter

# Configurazione del logging
# Imposta il livello di logging e il formato
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Directory di base
BASE_DIR = Path('/workspace')

# Directory per i risultati e i checkpoint
RESULTS_DIR = BASE_DIR / 'results'
TENSORBOARD_DIR = BASE_DIR / 'runs' / 'tensorboard'

# Verifico che le directory esistano
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_DIR, exist_ok=True)

# Funzione per caricare un file YAML
def load_yaml(file_path):
    """Carica un singolo file YAML."""
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    logging.info(f"Configurazione caricata: {file_path}")
    return data

# Salva i dati di produzione di CO2 in un file locale
def save_co2_data(co2_data, run_results_dir):
    """Salva i dati di produzione di CO2 in un file locale."""
    co2_file = run_results_dir / 'co2_production.txt'
    with open(co2_file, 'a') as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - CO2: {co2_data} kg\n")
    logging.info(f"CO2 salvato in {co2_file}")

# Configura TensorBoard per il monitoraggio dell'addestramento
def setup_tensorboard(run_name):
    """Configura TensorBoard per il monitoraggio dell'addestramento."""
    tb_log_dir = TENSORBOARD_DIR / run_name
    os.makedirs(tb_log_dir, exist_ok=True)
    
    # Crea un SummaryWriter per TensorBoard
    writer = SummaryWriter(log_dir=str(tb_log_dir))
    
    logging.info(f"TensorBoard configurato per {run_name}")
    logging.info(f"Log directory: {tb_log_dir}")
    logging.info(f"Per visualizzare TensorBoard, esegui: tensorboard --logdir={TENSORBOARD_DIR} --host=0.0.0.0 --port=6006")
    
    return writer, tb_log_dir

# Log delle metriche su TensorBoard
def log_to_tensorboard(trainer, writer):
    """Logga le metriche di training su TensorBoard."""
    try:
        epoch = trainer.epoch
        
        # Log delle metriche di training
        if hasattr(trainer, 'loss') and trainer.loss is not None:
            writer.add_scalar('Loss/Train', trainer.loss, epoch)
        
        # Log delle metriche di validazione se disponibili
        if hasattr(trainer, 'metrics') and trainer.metrics is not None:
            metrics = trainer.metrics
            
            # Metriche comuni YOLO
            metric_mappings = {
                'mAP50': 'Metrics/mAP@0.5',
                'mAP50-95': 'Metrics/mAP@0.5:0.95',
                'precision': 'Metrics/Precision',
                'recall': 'Metrics/Recall',
                'f1': 'Metrics/F1-Score',
                'val/box_loss': 'Loss/Val_Box',
                'val/cls_loss': 'Loss/Val_Class',
                'val/dfl_loss': 'Loss/Val_DFL'
            }
            
            for metric_key, tb_key in metric_mappings.items():
                if metric_key in metrics:
                    writer.add_scalar(tb_key, metrics[metric_key], epoch)
        
        # Log del learning rate
        if hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
            for i, param_group in enumerate(trainer.optimizer.param_groups):
                writer.add_scalar(f'Learning_Rate/Group_{i}', param_group['lr'], epoch)
                
        logging.info(f"Metriche epoch {epoch} loggate su TensorBoard")
        
    except Exception as e:
        logging.warning(f"Errore nel logging TensorBoard per epoch {trainer.epoch}: {e}")

# Test del modello
def test_model(model, run_name):
    """Esegue il test del modello sul dataset di test."""
    logging.info("Inizio del test del modello...")
    try:
        # Esegui il test utilizzando il set di test definito nel dataset YAML
        results = model.val(
            split="test",
            save_txt=True,
            save_json=True,
            save_conf=True,
            name=f"{run_name}_test",
            verbose=True,
            plots=True
        )
        
    except Exception as e:
        logging.error(f"Errore durante il test del modello: {e}")
        raise e

# Esegue l'addestramento del modello YOLO con le configurazioni specificate
def train_model(model_config_path, dataset_config_path, pretrained_weights_path = None, tracker=None):
    """Esegue l'addestramento del modello YOLO con le configurazioni specificate."""
    try:
        varResume = True  # Variabile per il resume dell'addestramento
        model_config = load_yaml(model_config_path)
        dataset_config = load_yaml(dataset_config_path)

        # Configura i nomi per il salvataggio dei risultati
        config_name = Path(model_config_path).stem
        dataset_name = Path(dataset_config_path).stem
        run_name = f"{config_name}_{dataset_name}"
        run_results_dir = RESULTS_DIR / run_name
        os.makedirs(run_results_dir, exist_ok=True)

        # Configura TensorBoard
        tb_writer, tb_log_dir = setup_tensorboard(run_name)

        try:
            # Inizializza il tracker di CO2
            tracker = EmissionsTracker(project_name=f"YOLO_{run_name}", output_dir=str(run_results_dir))
            tracker.start()
        except Exception as e:
            logging.error(f" 1 Errore durante l'addestramento di {run_name}: {e}")

        def check_for_test_tag(name: str):
            return "Test" in name or "test" in name

        # combined_config = {**model_config, **dataset_config}

        # Verifica e imposta i tag
        if check_for_test_tag(run_name):
            wandb.init(
                project="results",
                name=f"{run_name}",
                tags=["test"]
            )
        else:
            wandb.init(
                project="results",
                name=f"{run_name}",
            )

        # Carica il modello con Ultralytics
        model_architecture_path = str(model_config_path)

        if pretrained_weights_path:
            # Se i pesi preaddestrati sono forniti, carica il modello con i pesi preaddestrati e imposta varResume a False
            model = YOLO(model_architecture_path)
            model.load(str(pretrained_weights_path))
            varResume = False
            logging.info(f"Caricamento del modello con pesi preaddestrati da: {pretrained_weights_path} varResume={varResume}")
        else:
            # Se non sono forniti pesi preaddestrati, carica il modello senza pesi
            logging.info(f"Caricamento del modello senza pesi preaddestrati da: {model_architecture_path} varResume={varResume}")
            model = YOLO(model_architecture_path)
            varResume = True

        # Aggiungi il callback TensorBoard al modello
        model.add_callback('on_train_start', lambda trainer: logging.info(f"Training started - TensorBoard logs: {tb_log_dir}"))
        model.add_callback('on_epoch_end', lambda trainer: log_to_tensorboard(trainer, tb_writer))

        try:
            model.train(
                data=str(dataset_config_path),
                name=f"{run_name}",
                resume=varResume,
                **model_config.get('training', {})
            )
        except Exception as e:
            logging.error(f" 3 Errore durante l'addestramento di {run_name}: {e}")

        # Chiudi il TensorBoard writer
        tb_writer.close()

        try:
            # Termina il tracker di CO2 e ottieni i dati
            emissions = tracker.stop()
            logging.info(f"Emissioni di CO2 per {run_name}: {emissions} kg")

            # Salva i dati di CO2 in un file locale
            save_co2_data(emissions, run_results_dir)
        except Exception as e:
            logging.error(f" 5 Errore durante l'addestramento di {run_name}: {e}")
            raise e
        
        try:
            # Test del modello
            test_model(model, run_name)

        except Exception as e:
            logging.error(f" 6 Errore durante l'addestramento di {run_name}: {e}")
            tb_writer.close()
            wandb.finish()
            raise e

        finally:
            try:
                tb_writer.close()
                wandb.finish()
            except Exception as e:
                logging.error(f"6.5 wandb.finish() or tb_writer.close() {e}")

    except Exception as e:
        logging.error(f" 7 Errore durante l'addestramento di {run_name}: {e}")
        if 'tb_writer' in locals():
            tb_writer.close()
        wandb.finish()
        raise e

# Funzione principale per eseguire l'addestramento
def main():
    """Funzione principale per eseguire l'addestramento."""
    parser = argparse.ArgumentParser(description='Script per l\'addestramento di YOLO con Ultralytics')
    parser.add_argument('--config', type=str, required=True, help='Percorso al file YAML di configurazione del modello')
    parser.add_argument('--dataset', type=str, required=True, help='Percorso al file YAML di configurazione del dataset')
    parser.add_argument('--weights', type=str, help='Percorso al file dei pesi preaddestrati')

    args = parser.parse_args()

    model_config_path = Path(args.config)
    if not model_config_path.is_absolute():
        model_config_path = BASE_DIR / model_config_path

    dataset_config_path = Path(args.dataset)
    if not dataset_config_path.is_absolute():
        dataset_config_path = BASE_DIR / dataset_config_path

    pretrained_weights_path = Path(args.weights) if args.weights else None
    if pretrained_weights_path and not pretrained_weights_path.is_absolute():
        pretrained_weights_path = BASE_DIR / pretrained_weights_path

    if not model_config_path.exists():
        logging.error(f"File di configurazione del modello non trovato: {model_config_path}")
        sys.exit(1)

    if not dataset_config_path.exists():
        logging.error(f"File di configurazione del dataset non trovato: {dataset_config_path}")
        sys.exit(1)

    try:
        logging.info(
            f"Inizio dell'addestramento per la configurazione: {model_config_path} con dataset: {dataset_config_path}")
        if pretrained_weights_path:
            logging.info(f"Utilizzo dei pesi preaddestrati: {pretrained_weights_path}")

        train_model(model_config_path, dataset_config_path, pretrained_weights_path)
        logging.info(f"Addestramento completato per: {model_config_path}")

    except Exception as e:
        logging.error(f"Addestramento interrotto per {model_config_path}: {e}")

if __name__ == "__main__":
    main()