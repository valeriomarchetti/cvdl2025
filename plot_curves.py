#!/usr/bin/env python3
"""
Script per visualizzare le curve salvate dal export_metrics.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob

def load_curves_data(curves_file):
    """Carica i dati delle curve da file JSON o NPZ"""
    if curves_file.endswith('.json'):
        with open(curves_file, 'r') as f:
            return json.load(f)
    elif curves_file.endswith('.npz'):
        return dict(np.load(curves_file))
    else:
        raise ValueError("Formato file non supportato. Usa .json o .npz")

def plot_curves(curves_data, output_dir=None):
    """Plotta le curve P-R, F1, ecc."""
    experiment_name = curves_data.get('experiment', 'unknown')
    
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    
    # Crea figure per le diverse curve
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Curves for {experiment_name}', fontsize=16, fontweight='bold')
    
    # P-R Curve
    if 'p_curve' in curves_data and 'r_curve' in curves_data:
        ax = axes[0, 0]
        p_curve = np.array(curves_data['p_curve'])
        r_curve = np.array(curves_data['r_curve'])
        
        if p_curve.ndim > 1:
            p_curve = p_curve[0]  # Prendi la prima classe se multidimensionale
        if r_curve.ndim > 1:
            r_curve = r_curve[0]
            
        ax.plot(r_curve, p_curve, 'b-', linewidth=2, label='P-R Curve')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # F1 Curve
    if 'f1_curve' in curves_data:
        ax = axes[0, 1]
        f1_curve = np.array(curves_data['f1_curve'])
        
        if f1_curve.ndim > 1:
            f1_curve = f1_curve[0]
            
        # Genera threshold values (tipicamente da 0 a 1)
        thresholds = np.linspace(0, 1, len(f1_curve))
        ax.plot(thresholds, f1_curve, 'g-', linewidth=2, label='F1 Score')
        ax.set_xlabel('Confidence Threshold')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 Score vs Confidence Threshold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Precision vs Confidence
    if 'p_curve' in curves_data:
        ax = axes[1, 0]
        p_curve = np.array(curves_data['p_curve'])
        
        if p_curve.ndim > 1:
            p_curve = p_curve[0]
            
        thresholds = np.linspace(0, 1, len(p_curve))
        ax.plot(thresholds, p_curve, 'r-', linewidth=2, label='Precision')
        ax.set_xlabel('Confidence Threshold')
        ax.set_ylabel('Precision')
        ax.set_title('Precision vs Confidence Threshold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Recall vs Confidence
    if 'r_curve' in curves_data:
        ax = axes[1, 1]
        r_curve = np.array(curves_data['r_curve'])
        
        if r_curve.ndim > 1:
            r_curve = r_curve[0]
            
        thresholds = np.linspace(0, 1, len(r_curve))
        ax.plot(thresholds, r_curve, 'orange', linewidth=2, label='Recall')
        ax.set_xlabel('Confidence Threshold')
        ax.set_ylabel('Recall')
        ax.set_title('Recall vs Confidence Threshold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    
    # Salva il plot se specificato
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_filename = f"{experiment_name}_curves_plot.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot salvato in: {plot_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot curves from exported metrics')
    parser.add_argument('--curves_file', type=str, help='Path to curves JSON or NPZ file')
    parser.add_argument('--curves_dir', type=str, default='/workspace/metrics/curves', 
                       help='Directory containing curves files')
    parser.add_argument('--output_dir', type=str, default='/workspace/metrics/plots',
                       help='Directory to save plots')
    parser.add_argument('--experiment', type=str, help='Specific experiment name to plot')
    
    args = parser.parse_args()
    
    if args.curves_file:
        # Plot singolo file
        curves_data = load_curves_data(args.curves_file)
        plot_curves(curves_data, args.output_dir)
    
    elif args.experiment:
        # Plot esperimento specifico
        pattern = os.path.join(args.curves_dir, f"{args.experiment}_curves.json")
        files = glob.glob(pattern)
        if files:
            curves_data = load_curves_data(files[0])
            plot_curves(curves_data, args.output_dir)
        else:
            print(f"❌ Nessun file trovato per esperimento: {args.experiment}")
    
    else:
        # Plot tutti i file nella directory
        pattern = os.path.join(args.curves_dir, "*_curves.json")
        files = glob.glob(pattern)
        
        if not files:
            print(f"❌ Nessun file di curve trovato in: {args.curves_dir}")
            return
        
        print(f"📊 Trovati {len(files)} file di curve")
        for file_path in files:
            print(f"\n--- Plotting {os.path.basename(file_path)} ---")
            try:
                curves_data = load_curves_data(file_path)
                plot_curves(curves_data, args.output_dir)
            except Exception as e:
                print(f"❌ Errore nel plot di {file_path}: {e}")

if __name__ == "__main__":
    main()
