from ultralytics import YOLO
import os
import csv
import datetime
import argparse
import numpy as np
import json

# === PARSING ARGOMENTI DA RIGA DI COMANDO ===
parser = argparse.ArgumentParser(description='Export YOLO model validation metrics')
parser.add_argument('--weights', type=str, required=True, help='Path to model weights file (.pt)')
parser.add_argument('--dataset', type=str, required=True, help='Path to dataset YAML configuration file')
parser.add_argument('--modelname', type=str, required=True, help='Model name to use as prefix for experiment name')
args = parser.parse_args()

# === CONFIGURAZIONE ===
# Percorso ai pesi del modello addestrato
weights_path = args.weights

# YAML del dataset
data_yaml = args.dataset

# Nome del modello
model_name = args.modelname

# Nome esperimento (con prefisso modelname)
experiment_name = f'{model_name}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'

# File CSV dove loggare le metriche
csv_output_path = '/workspace/metrics/val_metrics_log.csv'

# Cartella per salvare le curve
curves_output_dir = '/workspace/metrics/curves'
os.makedirs(curves_output_dir, exist_ok=True)

# === CARICAMENTO MODELLO ===
model = YOLO(weights_path)

# === VALUTAZIONE SUL SET DI TEST ===
results = model.val(data=data_yaml, split='test', save_json=True, save_hybrid=True)

# === ESTRAZIONE METRICHE ===
metrics = {
    'experiment': experiment_name,
    'precision': float(results.box.mp),  # Mean precision
    'recall': float(results.box.mr),     # Mean recall
    'f1': float(results.box.f1[0]) if len(results.box.f1) > 0 else 0.0,  # F1 score
    'map50': float(results.box.map50),   # mAP@0.5
    'map75': float(results.box.map75),   # mAP@0.75
    'map': float(results.box.map),       # mAP@[.5:.95]
    'inference_time': results.speed['inference'],
    'nms_time': results.speed['postprocess']  # Il tempo di postprocess include NMS
}

# === SALVATAGGIO METRICHE IN CSV ===
file_exists = os.path.isfile(csv_output_path)

with open(csv_output_path, mode='a', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=metrics.keys())
    if not file_exists:
        writer.writeheader()
    writer.writerow(metrics)

print(f"✅ Metriche salvate su: {csv_output_path}")

# === SALVATAGGIO CURVE ===
# Apri file di debug per salvare tutto l'output
debug_file_path = os.path.join('/workspace', f"{experiment_name}_debug.txt")
debug_file = open(debug_file_path, 'w')

def debug_print(text):
    """Stampa sia su console che su file di debug"""
    print(text)
    debug_file.write(text + '\n')
    debug_file.flush()

debug_print("\n=== DEBUG STRUTTURA CURVE ===")

# Debug approfondito degli oggetti curves
curve_attributes = ['f1_curve', 'p_curve', 'r_curve', 'curves', 'curves_results']

for attr in curve_attributes:
    if hasattr(results.box, attr):
        value = getattr(results.box, attr)
        debug_print(f"\n--- {attr} ---")
        debug_print(f"Tipo: {type(value)}")
        
        # Controlla se è stampabile direttamente
        try:
            if hasattr(value, 'shape') and value.shape[0] * value.shape[1] <= 50:
                debug_print(f"Valore completo: {value}")
            elif isinstance(value, list) and len(value) <= 10:
                debug_print(f"Valore completo: {value}")
            else:
                debug_print(f"Valore (troppo grande per stampa completa)")
        except:
            debug_print(f"Errore nella stampa del valore")
        
        if hasattr(value, 'shape'):
            debug_print(f"Shape: {value.shape}")
        if hasattr(value, '__len__'):
            try:
                debug_print(f"Lunghezza: {len(value)}")
            except:
                pass
        
        # Se è un array numpy, mostra primi e ultimi elementi
        if isinstance(value, np.ndarray):
            debug_print(f"Dtype: {value.dtype}")
            if value.size > 0:
                debug_print(f"Min: {value.min()}, Max: {value.max()}")
                if value.size <= 10:
                    debug_print(f"Tutti i valori: {value}")
                else:
                    debug_print(f"Primi 5: {value.flat[:5]}")
                    debug_print(f"Ultimi 5: {value.flat[-5:]}")
        
        # Se è una lista, mostra info sulla struttura
        elif isinstance(value, list):
            if len(value) > 0:
                debug_print(f"Primo elemento tipo: {type(value[0])}")
                if hasattr(value[0], '__len__'):
                    try:
                        debug_print(f"Primo elemento lunghezza: {len(value[0])}")
                        # Se il primo elemento è una lista, mostra alcuni valori
                        if isinstance(value[0], list) and len(value[0]) <= 20:
                            debug_print(f"Primo elemento contenuto: {value[0]}")
                        elif isinstance(value[0], list):
                            debug_print(f"Primo elemento primi 5: {value[0][:5]}")
                    except:
                        pass
        
        debug_print("-" * 30)

debug_print("\n=== FINE DEBUG STRUTTURA ===")

debug_print("\n=== SALVATAGGIO CURVE ===")

# Dizionario per le curve con controlli di sicurezza migliorati
curves_data = {}

def safe_convert_to_serializable(obj, name="unknown"):
    """Converte oggetti numpy/tensor in formato serializzabile con debug """
    debug_print(f"\n--- Conversione {name} ---")
    debug_print(f"Tipo originale: {type(obj)}")
    
    if obj is None:
        debug_print("Oggetto è None")
        return None
    
    if isinstance(obj, np.ndarray):
        debug_print(f"Array numpy shape: {obj.shape}, dtype: {obj.dtype}")
        if obj.size == 0:
            debug_print("Array vuoto")
            return []
        # Converti in lista Python ricorsivamente
        try:
            result = obj.tolist()
            debug_print(f"Convertito in lista")
            # Verifica che sia effettivamente serializzabile
            json.dumps(result)  # Test di serializzazione
            return result
        except Exception as e:
            debug_print(f"Errore nella conversione tolist(): {e}")
            return str(obj)
    
    elif isinstance(obj, (list, tuple)):
        debug_print(f"Lista/tupla di lunghezza: {len(obj)}")
        if len(obj) == 0:
            return []
        
        # Controlla il tipo degli elementi
        first_element = obj[0]
        debug_print(f"Primo elemento tipo: {type(first_element)}")
        
        # Ricorsivamente converti elementi se necessario
        result = []
        for i, item in enumerate(obj):
            debug_print(f"  Elaborando elemento {i}: tipo {type(item)}")
            
            try:
                if isinstance(item, np.ndarray):
                    debug_print(f"    Array numpy shape: {item.shape}")
                    converted_item = item.tolist()
                    # Test di serializzazione per l'array convertito
                    json.dumps(converted_item)
                    debug_print(f"    Array convertito con successo")
                elif isinstance(item, (list, tuple)):
                    # Ricorsione per liste annidate
                    converted_item = safe_convert_to_serializable(item, f"{name}_item_{i}")
                elif isinstance(item, (str, int, float, bool)):
                    converted_item = item
                    debug_print(f"    Tipo primitivo: {converted_item}")
                elif hasattr(item, 'tolist'):
                    converted_item = item.tolist()
                    json.dumps(converted_item)
                else:
                    converted_item = str(item)
                    debug_print(f"    Convertito in stringa: {converted_item}")
                
                result.append(converted_item)
                
            except Exception as e:
                debug_print(f"  Errore conversione elemento {i}: {e}")
                # Fallback: converti in stringa
                result.append(str(item))
        
        debug_print(f"Lista convertita con {len(result)} elementi")
        
        # Test finale di serializzazione della lista completa
        try:
            json.dumps(result)
            debug_print("Lista finale serializzabile")
            return result
        except Exception as e:
            debug_print(f"Errore serializzazione lista finale: {e}")
            # Fallback: converti tutto in stringhe
            return [str(item) for item in result]
    
    elif hasattr(obj, 'tolist'):
        debug_print("Ha metodo tolist()")
        try:
            result = obj.tolist()
            json.dumps(result)  # Test serializzazione
            return result
        except Exception as e:
            debug_print(f"Errore tolist(): {e}")
            return str(obj)
    
    elif isinstance(obj, (int, float, str, bool)):
        debug_print(f"Tipo primitivo: {type(obj)}")
        return obj
    
    else:
        debug_print(f"Tipo non gestito: {type(obj)}, converting to string")
        return str(obj)

# Estrai le curve disponibili con conversione sicura
if hasattr(results.box, 'f1_curve') and results.box.f1_curve is not None:
    curves_data['f1_curve'] = safe_convert_to_serializable(results.box.f1_curve, 'f1_curve')
    debug_print("✅ F1 curve estratta")

if hasattr(results.box, 'p_curve') and results.box.p_curve is not None:
    curves_data['p_curve'] = safe_convert_to_serializable(results.box.p_curve, 'p_curve')
    debug_print("✅ Precision curve estratta")

if hasattr(results.box, 'r_curve') and results.box.r_curve is not None:
    curves_data['r_curve'] = safe_convert_to_serializable(results.box.r_curve, 'r_curve')
    debug_print("✅ Recall curve estratta")

if hasattr(results.box, 'curves') and results.box.curves is not None:
    curves_data['curves'] = safe_convert_to_serializable(results.box.curves, 'curves')
    debug_print("✅ Curves estratte")

if hasattr(results.box, 'curves_results') and results.box.curves_results is not None:
    debug_print("\n=== ANALISI CURVES_RESULTS ===")
    curves_results_raw = results.box.curves_results
    if len(curves_results_raw) >= 4:
        debug_print(f"curves_results ha {len(curves_results_raw)} elementi:")
        debug_print(f"  Elemento 0 (threshold): shape {curves_results_raw[0].shape if hasattr(curves_results_raw[0], 'shape') else 'N/A'}")
        debug_print(f"  Elemento 1 (valori): shape {curves_results_raw[1].shape if hasattr(curves_results_raw[1], 'shape') else 'N/A'}")
        debug_print(f"  Elemento 2: {curves_results_raw[2]}")
        debug_print(f"  Elemento 3: {curves_results_raw[3]}")
        
        # Estrai i dati in formato strutturato usando la funzione safe
        structured_curves = {
            'thresholds': safe_convert_to_serializable(curves_results_raw[0], 'thresholds'),
            'values': safe_convert_to_serializable(curves_results_raw[1], 'values'),
            'y_label': str(curves_results_raw[2]) if len(curves_results_raw) > 2 else 'Unknown',
            'x_label': str(curves_results_raw[3]) if len(curves_results_raw) > 3 else 'Unknown'
        }
        curves_data['curves_results_structured'] = structured_curves
        debug_print("✅ Curves results estratte in formato strutturato")
    
    # Mantieni anche il formato originale per compatibilità
    curves_data['curves_results'] = safe_convert_to_serializable(results.box.curves_results, 'curves_results')
    debug_print("✅ Curves results estratte (formato raw)")

else:
    debug_print("⚠️ curves_results non disponibile")

# Aggiungi metadati
curves_data['experiment'] = experiment_name
curves_data['model_name'] = model_name
curves_data['timestamp'] = datetime.datetime.now().isoformat()
curves_data['weights_path'] = weights_path
curves_data['dataset_path'] = data_yaml

# Salva le curve in formato JSON
curves_filename = f"{experiment_name}_curves.json"
curves_filepath = os.path.join(curves_output_dir, curves_filename)

# Controllo finale - verifica ogni chiave separatamente
debug_print("\n=== CONTROLLO FINALE ===")
clean_curves_data = {}

for key, value in curves_data.items():
    debug_print(f"Controllo {key}...")
    try:
        # Test di serializzazione per ogni chiave
        json.dumps(value)
        clean_curves_data[key] = value
        debug_print(f"  ✅ {key} serializzabile")
    except Exception as e:
        debug_print(f"  ❌ {key} non serializzabile: {e}")
        # Converte in un formato serializzabile
        try:
            converted_value = safe_convert_to_serializable(value, f"final_{key}")
            json.dumps(converted_value)
            clean_curves_data[key] = converted_value
            debug_print(f"  ✅ {key} convertito e ora serializzabile")
        except Exception as e2:
            debug_print(f"  ❌ {key} non salvabile: {e2}")
            # Fallback: salva solo info di base
            clean_curves_data[key] = str(value)[:200] + "..." if len(str(value)) > 200 else str(value)

try:
    # Test finale di serializzazione dell'intero dizionario
    json_str = json.dumps(clean_curves_data, indent=2)
    with open(curves_filepath, 'w') as f:
        f.write(json_str)
    debug_print(f"✅ Curve salvate con successo in: {curves_filepath}")
except Exception as e:
    debug_print(f"❌ Errore nel salvataggio finale: {e}")
    # Salva almeno i metadati
    try:
        minimal_data = {
            'experiment': experiment_name,
            'timestamp': datetime.datetime.now().isoformat(),
            'weights_path': weights_path,
            'dataset_path': data_yaml,
            'error': str(e),
            'keys_attempted': list(curves_data.keys())
        }
        with open(curves_filepath, 'w') as f:
            json.dump(minimal_data, f, indent=2)
        debug_print(f"⚠️ Salvati almeno i metadati in: {curves_filepath}")
    except Exception as e2:
        debug_print(f"❌ Errore anche nel salvataggio minimo: {e2}")

# Chiudi il file di debug
debug_file.close()
print(f"📋 Debug completo salvato in: {debug_file_path}")

# Salva anche in formato numpy per facilità di caricamento
numpy_filename = f"{experiment_name}_curves.npz"
numpy_filepath = os.path.join(curves_output_dir, numpy_filename)

try:
    # Prepara i dati per numpy (solo array numerici)
    numpy_data = {}
    for key, value in curves_data.items():
        if isinstance(value, (list, np.ndarray)) and key.endswith('_curve'):
            numpy_data[key] = np.array(value)
    
    if numpy_data:
        np.savez(numpy_filepath, **numpy_data)
        print(f"✅ Curve numpy salvate in: {numpy_filepath}")
except Exception as e:
    print(f"⚠️ Errore nel salvataggio numpy: {e}")

print("=" * 30)

# === STAMPA METRICHE ===
print("\n=== METRICHE ESTRATTE ===")
for key, value in metrics.items():
    print(f"{key}: {value}")
print("=" * 25)

print("✅ Script completato con successo!")
