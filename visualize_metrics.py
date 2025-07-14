#!/usr/bin/env python3
"""
Script avanzato per visualizzare e confrontare metriche di diversi esperimenti YOLO
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os
import glob
import argparse
from pathlib import Path

# Configurazione stile matplotlib
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")

class MetricsVisualizer:
    def __init__(self, csv_path, curves_dir=None, output_dir=None):
        self.csv_path = csv_path
        self.curves_dir = curves_dir or "/workspace/metrics/curves"
        self.output_dir = output_dir or "/workspace/metrics/plots"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Carica dati
        self.df = self.load_metrics_csv()
        self.curves_data = self.load_all_curves()
        
    def load_metrics_csv(self):
        """Carica il CSV delle metriche"""
        try:
            df = pd.read_csv(self.csv_path)
            print(f"✅ Caricati {len(df)} esperimenti dal CSV")
            return df
        except Exception as e:
            print(f"❌ Errore nel caricamento CSV: {e}")
            return pd.DataFrame()
    
    def load_all_curves(self):
        """Carica tutti i file JSON delle curve"""
        curves_data = {}
        if not os.path.exists(self.curves_dir):
            print(f"⚠️ Directory curve non trovata: {self.curves_dir}")
            return curves_data
            
        json_files = glob.glob(os.path.join(self.curves_dir, "*_curves.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    experiment_name = data.get('experiment', os.path.basename(json_file).replace('_curves.json', ''))
                    curves_data[experiment_name] = data
                    print(f"✅ Caricato {experiment_name}")
            except Exception as e:
                print(f"⚠️ Errore nel caricamento {json_file}: {e}")
        
        print(f"📊 Caricati {len(curves_data)} file di curve")
        return curves_data
    
    def plot_metrics_comparison(self):
        """Crea grafici di confronto delle metriche principali"""
        if self.df.empty:
            print("❌ Nessun dato da plottare")
            return
            
        # Metriche da confrontare
        metrics = ['precision', 'recall', 'f1', 'map50', 'map75', 'map']
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.df)))
        
        # Crea subplot
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Confronto Metriche tra Esperimenti', fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Estrai nomi esperimenti (solo timestamp finale)
            experiment_labels = [exp.split('_')[0] for exp in self.df['experiment']]
            values = self.df[metric].values
            
            # Bar plot con colori diversi
            bars = ax.bar(range(len(experiment_labels)), values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Personalizzazione
            ax.set_xlabel('Esperimento (Timestamp)', fontweight='bold')
            ax.set_ylabel(metric.upper(), fontweight='bold')
            ax.set_title(f'{metric.upper()} Comparison', fontsize=14, fontweight='bold')
            ax.set_xticks(range(len(experiment_labels)))
            ax.set_xticklabels(experiment_labels, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Aggiungi valori sopra le barre
            for j, (bar, value) in enumerate(zip(bars, values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + height*0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # Evidenzia il migliore
            best_idx = np.argmax(values)
            bars[best_idx].set_color('gold')
            bars[best_idx].set_edgecolor('red')
            bars[best_idx].set_linewidth(2)
        
        plt.tight_layout()
        
        # Salva il plot
        plot_path = os.path.join(self.output_dir, 'metrics_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Grafici metriche salvati in: {plot_path}")
        plt.show()
        
    def plot_performance_times(self):
        """Grafico dei tempi di inferenza e NMS"""
        if self.df.empty or 'inference_time' not in self.df.columns:
            print("⚠️ Dati dei tempi non disponibili")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Analisi Tempi di Performance', fontsize=16, fontweight='bold')
        
        experiment_labels = [exp.split('_')[0] for exp in self.df['experiment']]
        
        # Grafico tempi di inferenza
        ax1.bar(experiment_labels, self.df['inference_time'], color='skyblue', alpha=0.8, edgecolor='black')
        ax1.set_title('Tempo di Inferenza', fontweight='bold')
        ax1.set_ylabel('Tempo (ms)', fontweight='bold')
        ax1.set_xlabel('Esperimento', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Aggiungi valori
        for i, v in enumerate(self.df['inference_time']):
            ax1.text(i, v + v*0.01, f'{v:.2f}ms', ha='center', va='bottom', fontweight='bold')
        
        # Grafico tempi NMS
        ax2.bar(experiment_labels, self.df['nms_time'], color='lightcoral', alpha=0.8, edgecolor='black')
        ax2.set_title('Tempo NMS/Postprocess', fontweight='bold')
        ax2.set_ylabel('Tempo (ms)', fontweight='bold')
        ax2.set_xlabel('Esperimento', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Aggiungi valori
        for i, v in enumerate(self.df['nms_time']):
            ax2.text(i, v + v*0.01, f'{v:.2f}ms', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Salva il plot
        plot_path = os.path.join(self.output_dir, 'performance_times.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Grafici tempi salvati in: {plot_path}")
        plt.show()
    
    def plot_curves_comparison(self):
        """Confronta le curve P-R, F1, ecc. di diversi esperimenti"""
        if not self.curves_data:
            print("⚠️ Nessuna curva disponibile per il confronto")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Confronto Curve tra Esperimenti', fontsize=16, fontweight='bold')
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.curves_data)))
        
        for i, (exp_name, data) in enumerate(self.curves_data.items()):
            color = colors[i % len(colors)]
            label = exp_name.split('_')[0]  # Solo timestamp
            
            # F1 Curve
            if 'f1_curve' in data and data['f1_curve']:
                ax = axes[0, 0]
                f1_values = np.array(data['f1_curve'][0] if isinstance(data['f1_curve'][0], list) else data['f1_curve'])
                thresholds = np.linspace(0, 1, len(f1_values))
                ax.plot(thresholds, f1_values, label=label, color=color, linewidth=2)
                ax.set_title('F1 Score vs Confidence Threshold', fontweight='bold')
                ax.set_xlabel('Confidence Threshold')
                ax.set_ylabel('F1 Score')
                ax.grid(True, alpha=0.3)
                ax.legend()
            
            # Precision Curve
            if 'p_curve' in data and data['p_curve']:
                ax = axes[0, 1]
                p_values = np.array(data['p_curve'][0] if isinstance(data['p_curve'][0], list) else data['p_curve'])
                thresholds = np.linspace(0, 1, len(p_values))
                ax.plot(thresholds, p_values, label=label, color=color, linewidth=2)
                ax.set_title('Precision vs Confidence Threshold', fontweight='bold')
                ax.set_xlabel('Confidence Threshold')
                ax.set_ylabel('Precision')
                ax.grid(True, alpha=0.3)
                ax.legend()
            
            # Recall Curve
            if 'r_curve' in data and data['r_curve']:
                ax = axes[1, 0]
                r_values = np.array(data['r_curve'][0] if isinstance(data['r_curve'][0], list) else data['r_curve'])
                thresholds = np.linspace(0, 1, len(r_values))
                ax.plot(thresholds, r_values, label=label, color=color, linewidth=2)
                ax.set_title('Recall vs Confidence Threshold', fontweight='bold')
                ax.set_xlabel('Confidence Threshold')
                ax.set_ylabel('Recall')
                ax.grid(True, alpha=0.3)
                ax.legend()
            
            # Precision-Recall Curve
            if 'p_curve' in data and 'r_curve' in data and data['p_curve'] and data['r_curve']:
                ax = axes[1, 1]
                p_values = np.array(data['p_curve'][0] if isinstance(data['p_curve'][0], list) else data['p_curve'])
                r_values = np.array(data['r_curve'][0] if isinstance(data['r_curve'][0], list) else data['r_curve'])
                ax.plot(r_values, p_values, label=label, color=color, linewidth=2)
                ax.set_title('Precision-Recall Curve', fontweight='bold')
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.grid(True, alpha=0.3)
                ax.legend()
        
        plt.tight_layout()
        
        # Salva il plot
        plot_path = os.path.join(self.output_dir, 'curves_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Grafici curve salvati in: {plot_path}")
        plt.show()
    
    def generate_summary_report(self):
        """Genera un report riassuntivo testuale"""
        if self.df.empty:
            print("❌ Nessun dato per il report")
            return
            
        print("\n" + "="*60)
        print("REPORT RIASSUNTIVO ESPERIMENTI")
        print("="*60)
        
        print(f"\nNumero totale di esperimenti: {len(self.df)}")
        
        # Migliori performance per ogni metrica
        metrics = ['precision', 'recall', 'f1', 'map50', 'map75', 'map']
        
        print("\nMIGLIORI PERFORMANCE:")
        for metric in metrics:
            if metric in self.df.columns:
                best_idx = self.df[metric].idxmax()
                best_exp = self.df.loc[best_idx, 'experiment']
                best_value = self.df.loc[best_idx, metric]
                print(f"  {metric.upper():<12}: {best_value:.4f} ({best_exp.split('_')[0]})")
        
        # Statistiche tempi
        if 'inference_time' in self.df.columns:
            print(f"\nTEMPI:")
            print(f"  Inferenza media: {self.df['inference_time'].mean():.2f}ms")
            print(f"  Inferenza min:   {self.df['inference_time'].min():.2f}ms")
            print(f"  Inferenza max:   {self.df['inference_time'].max():.2f}ms")
            print(f"  NMS medio:       {self.df['nms_time'].mean():.2f}ms")
        
        # Tabella completa
        print(f"\nTABELLA COMPLETA:")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(self.df.to_string(index=False, float_format='%.4f'))
        
        print("="*60)
    
    def create_dashboard(self):
        """Crea una dashboard completa con tutti i grafici"""
        print("Generazione dashboard completa...")
        
        # Genera tutti i grafici
        self.plot_metrics_comparison()
        self.plot_performance_times()
        self.plot_curves_comparison()
        self.generate_summary_report()
        
        print(f"\nDashboard completa generata in: {self.output_dir}")
        print("File generati:")
        for file in os.listdir(self.output_dir):
            if file.endswith('.png'):
                print(f"  - {file}")

def main():
    parser = argparse.ArgumentParser(description='Visualizza e confronta metriche YOLO')
    parser.add_argument('--csv_path', type=str, default='/workspace/metrics/val_metrics_log.csv',
                       help='Path al file CSV delle metriche')
    parser.add_argument('--curves_dir', type=str, default='/workspace/metrics/curves',
                       help='Directory contenente i file JSON delle curve')
    parser.add_argument('--output_dir', type=str, default='/workspace/metrics/plots',
                       help='Directory di output per i grafici')
    parser.add_argument('--dashboard', action='store_true',
                       help='Genera dashboard completa')
    parser.add_argument('--metrics_only', action='store_true',
                       help='Solo grafici metriche')
    parser.add_argument('--curves_only', action='store_true',
                       help='Solo grafici curve')
    parser.add_argument('--report_only', action='store_true',
                       help='Solo report testuale')
    
    args = parser.parse_args()
    
    # Verifica che il file CSV esista
    if not os.path.exists(args.csv_path):
        print(f"❌ File CSV non trovato: {args.csv_path}")
        return
    
    # Crea visualizzatore
    visualizer = MetricsVisualizer(args.csv_path, args.curves_dir, args.output_dir)
    
    # Genera visualizzazioni in base ai parametri
    if args.dashboard:
        visualizer.create_dashboard()
    elif args.metrics_only:
        visualizer.plot_metrics_comparison()
        visualizer.plot_performance_times()
    elif args.curves_only:
        visualizer.plot_curves_comparison()
    elif args.report_only:
        visualizer.generate_summary_report()
    else:
        # Default: tutto
        visualizer.create_dashboard()

if __name__ == "__main__":
    main()
