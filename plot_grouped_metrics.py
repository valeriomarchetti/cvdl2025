#!/usr/bin/env python3
"""
Script per creare grafici raggruppati per tematica delle metriche YOLO
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def load_data(csv_path):
    """Carica i dati dal CSV"""
    try:
        df = pd.read_csv(csv_path)
        print(f"Caricati {len(df)} esperimenti")
        return df
    except Exception as e:
        print(f"Errore nel caricamento: {e}")
        return None

def create_grouped_plots(df, output_dir):
    """Crea grafici raggruppati per tematica"""
    
    # Crea directory di output
    os.makedirs(output_dir, exist_ok=True)
    
    # Estrai i nomi degli esperimenti (prima parte - nome modello)
    experiment_names = [exp.split('_')[0] for exp in df['experiment']]
    
    # Palette di colori per le metriche (non per gli esperimenti)
    metric_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    experiment_colors = plt.cm.Set3(np.linspace(0, 1, len(experiment_names)))
    
    # 1. METRICHE DI QUALITÀ
    quality_metrics = ['precision', 'recall', 'f1']
    if all(col in df.columns for col in quality_metrics):
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(experiment_names))
        width = 0.25
        
        for i, metric in enumerate(quality_metrics):
            values = df[metric].values
            bars = ax.bar(x + i*width, values, width, label=metric.upper(), 
                         color=metric_colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Aggiungi valori sulle barre
            for j, (bar, value) in enumerate(zip(bars, values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Esperimenti (Nome Modello)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Valore Metrica', fontweight='bold', fontsize=12)
        ax.set_title('METRICHE DI QUALITÀ: Precision, Recall, F1', fontweight='bold', fontsize=14)
        ax.set_xticks(x + width)
        ax.set_xticklabels(experiment_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'quality_metrics.png'), dpi=300, bbox_inches='tight')
        print("✓ Salvato: quality_metrics.png")
        plt.show()
    
    # 2. METRICHE mAP
    #map_metrics = ['map50', 'map']
    map_metrics = ['map50', 'map75', 'map']
    if all(col in df.columns for col in map_metrics):
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(experiment_names))
        width = 0.25
        
        for i, metric in enumerate(map_metrics):
            values = df[metric].values
            bars = ax.bar(x + i*width, values, width, label=metric.upper(), 
                         color=metric_colors[i+3], alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Aggiungi valori sulle barre
            for j, (bar, value) in enumerate(zip(bars, values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Esperimenti (Nome Modello)', fontweight='bold', fontsize=12)
        ax.set_ylabel('mAP Score', fontweight='bold', fontsize=12)
        ax.set_title('METRICHE mAP: mAP@0.5, mAP@0.75, mAP@0.5:0.95', fontweight='bold', fontsize=14)
        #ax.set_title('METRICHE mAP: mAP@0.5, mAP@0.5:0.95', fontweight='bold', fontsize=14)
        ax.set_xticks(x + width)
        ax.set_xticklabels(experiment_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'map_metrics.png'), dpi=300, bbox_inches='tight')
        print("✓ Salvato: map_metrics.png")
        plt.show()
    
    # 3. TEMPI DI PERFORMANCE
    time_metrics = ['inference_time', 'nms_time']
    if all(col in df.columns for col in time_metrics):
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(experiment_names))
        width = 0.35
        
        for i, metric in enumerate(time_metrics):
            values = df[metric].values
            label = 'Inference Time' if metric == 'inference_time' else 'NMS Time'
            bars = ax.bar(x + i*width, values, width, label=label, 
                         color=metric_colors[i+6], alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Aggiungi valori sulle barre
            for j, (bar, value) in enumerate(zip(bars, values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + height*0.02,
                       f'{value:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Esperimenti (Nome Modello)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Tempo (ms)', fontweight='bold', fontsize=12)
        ax.set_title('TEMPI DI PERFORMANCE: Inference e NMS', fontweight='bold', fontsize=14)
        ax.set_xticks(x + width/2)
        ax.set_xticklabels(experiment_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_times.png'), dpi=300, bbox_inches='tight')
        print("✓ Salvato: performance_times.png")
        plt.show()
    
    # 4. OVERVIEW COMPLESSIVO (tutte le metriche principali)
    overview_metrics = ['precision', 'recall', 'f1', 'map50', 'map']
    if all(col in df.columns for col in overview_metrics):
        fig, ax = plt.subplots(figsize=(16, 10))
        
        x = np.arange(len(experiment_names))
        width = 0.15
        
        for i, metric in enumerate(overview_metrics):
            values = df[metric].values
            bars = ax.bar(x + i*width, values, width, label=metric.upper(), 
                         color=metric_colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Aggiungi valori sulle barre (solo ogni 2 bar per evitare sovrapposizioni)
            if i % 2 == 0:
                for j, (bar, value) in enumerate(zip(bars, values)):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                           f'{value:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax.set_xlabel('Esperimenti (Nome Modello)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Score', fontweight='bold', fontsize=12)
        ax.set_title('OVERVIEW COMPLETO: Tutte le Metriche Principali', fontweight='bold', fontsize=16)
        ax.set_xticks(x + width*2)
        ax.set_xticklabels(experiment_names, rotation=45, ha='right')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'overview_all_metrics.png'), dpi=300, bbox_inches='tight')
        print("✓ Salvato: overview_all_metrics.png")
        plt.show()

def generate_stats_report(df):
    """Genera un report statistico semplificato"""
    print("\n" + "="*50)
    print("STATISTICHE RIASSUNTIVE")
    print("="*50)
    
    # Metriche principali
    main_metrics = ['precision', 'recall', 'f1', 'map50', 'map75', 'map']
    
    print("\nMEDIE:")
    for metric in main_metrics:
        if metric in df.columns:
            mean_val = df[metric].mean()
            std_val = df[metric].std()
            print(f"  {metric.upper():<12}: {mean_val:.4f} ± {std_val:.4f}")
    
    print("\nMIGLIORI VALORI:")
    for metric in main_metrics:
        if metric in df.columns:
            best_idx = df[metric].idxmax()
            best_exp = df.loc[best_idx, 'experiment'].split('_')[0]
            best_val = df.loc[best_idx, metric]
            print(f"  {metric.upper():<12}: {best_val:.4f} (exp: {best_exp})")
    
    if 'inference_time' in df.columns:
        print(f"\nTEMPI:")
        print(f"  Inference: {df['inference_time'].mean():.2f} ± {df['inference_time'].std():.2f} ms")
        print(f"  NMS:       {df['nms_time'].mean():.2f} ± {df['nms_time'].std():.2f} ms")

def main():
    parser = argparse.ArgumentParser(description='Grafici raggruppati per tematica delle metriche YOLO')
    parser.add_argument('--csv', type=str, default='/workspace/metrics/val_metrics_log.csv',
                       help='Path del file CSV')
    parser.add_argument('--output', type=str, default='/workspace/metrics/grouped_plots',
                       help='Directory di output')
    
    args = parser.parse_args()
    
    # Carica dati
    df = load_data(args.csv)
    if df is None:
        return
    
    # Genera grafici
    create_grouped_plots(df, args.output)
    
    # Report statistico
    generate_stats_report(df)
    
    print(f"\nTutti i grafici salvati in: {args.output}")

if __name__ == "__main__":
    main()
