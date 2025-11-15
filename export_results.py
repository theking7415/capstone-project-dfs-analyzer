"""
Export RDFS pickle results to readable formats (CSV and JSON)
"""

import pickle
import json
import csv
import os
from pathlib import Path
from myrdfs import get_summary_stats

def read_pickle(fname):
    """Load graph and statistics from pickle file"""
    with open(fname, "rb") as f:
        return pickle.load(f)

def export_to_csv(graph, dist_stats, summary_stats, output_file):
    """Export results to CSV format"""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['Vertex', 'Mean_Discovery', 'Std_Dev', 'Min', 'Max', 'Median', 'Samples'])
        
        # Data rows sorted by vertex
        for vertex in sorted(summary_stats.keys()):
            stats = summary_stats[vertex]
            raw_data = dist_stats[vertex]
            
            writer.writerow([
                str(vertex),
                f"{stats.mean:.4f}",
                f"{(stats.variance ** 0.5):.4f}",
                f"{min(raw_data):.4f}",
                f"{max(raw_data):.4f}",
                f"{sorted(raw_data)[len(raw_data)//2]:.4f}",
                len(raw_data)
            ])

def export_to_json(graph, dist_stats, summary_stats, output_file):
    """Export results to JSON format"""
    data = {
        'graph': {
            'type': graph.desc(),
            'num_vertices': graph.number_vertices(),
        },
        'statistics': {}
    }
    
    # Add per-vertex statistics
    for vertex in sorted(summary_stats.keys(), key=str):
        stats = summary_stats[vertex]
        raw_data = dist_stats[vertex]
        
        data['statistics'][str(vertex)] = {
            'mean': float(stats.mean),
            'std_dev': float(stats.variance ** 0.5),
            'variance': float(stats.variance),
            'min': float(min(raw_data)),
            'max': float(max(raw_data)),
            'median': float(sorted(raw_data)[len(raw_data)//2]),
            'num_samples': len(raw_data)
        }
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

def export_to_txt(graph, dist_stats, summary_stats, output_file):
    """Export results to readable text format"""
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"RDFS Results for {graph.desc().upper()}\n")
        f.write(f"Total Vertices: {graph.number_vertices()}\n")
        f.write("=" * 80 + "\n\n")
        
        for vertex in sorted(summary_stats.keys()):
            stats = summary_stats[vertex]
            raw_data = dist_stats[vertex]
            
            f.write(f"Vertex: {vertex}\n")
            f.write(f"  Samples: {len(raw_data)}\n")
            f.write(f"  Mean Discovery Number: {stats.mean:.4f}\n")
            f.write(f"  Std Deviation: {(stats.variance ** 0.5):.4f}\n")
            f.write(f"  Min: {min(raw_data):.4f}\n")
            f.write(f"  Max: {max(raw_data):.4f}\n")
            f.write(f"  Median: {sorted(raw_data)[len(raw_data)//2]:.4f}\n")
            f.write("\n")

def main():
    """Export all pickle files to CSV, JSON, and TXT"""
    
    data_dir = "data"
    output_dir = "results"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Find all pickle files
    pickle_files = sorted(Path(data_dir).glob("rdfs-*.pickle"))
    
    if not pickle_files:
        print("No pickle files found in data/ directory")
        return
    
    print("=" * 80)
    print("Exporting RDFS Results to Readable Formats")
    print("=" * 80)
    
    for pickle_file in pickle_files:
        print(f"\nProcessing: {pickle_file.name}")
        
        try:
            # Load data
            graph, dist_stats = read_pickle(str(pickle_file))
            summary_stats = get_summary_stats(dist_stats)
            
            # Get base filename without extension
            base_name = pickle_file.stem
            
            # Export to CSV
            csv_file = f"{output_dir}/{base_name}.csv"
            export_to_csv(graph, dist_stats, summary_stats, csv_file)
            print(f"  ✓ Exported to CSV: {csv_file}")
            
            # Export to JSON
            json_file = f"{output_dir}/{base_name}.json"
            export_to_json(graph, dist_stats, summary_stats, json_file)
            print(f"  ✓ Exported to JSON: {json_file}")
            
            # Export to TXT
            txt_file = f"{output_dir}/{base_name}.txt"
            export_to_txt(graph, dist_stats, summary_stats, txt_file)
            print(f"  ✓ Exported to TXT: {txt_file}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\n" + "=" * 80)
    print(f"All results exported to '{output_dir}/' directory")
    print("=" * 80)

if __name__ == "__main__":
    main()
