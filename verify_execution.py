import json
import sys
import os

def verify():
    notebook_path = "credit_scoring_optimization.ipynb"
    if not os.path.exists(notebook_path):
        print(f"Error: {notebook_path} does not exist!")
        sys.exit(1)
        
    print(f"Reading notebook '{notebook_path}'...")
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    code_cells = [cell["source"] for cell in nb["cells"] if cell["cell_type"] == "code"]
    print(f"Found {len(code_cells)} code cells.")
    
    # Concatenate all code cells
    full_code = ""
    # Inject headless matplotlib backend to prevent blocking on plt.show()
    full_code += "import matplotlib\nmatplotlib.use('Agg')\n"
    
    for idx, cell_source in enumerate(code_cells):
        full_code += f"\n# --- CODE CELL {idx+1} ---\n"
        full_code += "".join(cell_source)
        
    print("Executing all code cells in sequence...")
    print("=" * 60)
    
    local_env = {
        'display': print
    }
    try:
        exec(full_code, globals(), local_env)
        print("=" * 60)
        print("SUCCESS: Notebook executed from end-to-end without errors!")
        
        # Verify generated plots
        expected_plots = [
            'class_distribution.png',
            'numerical_features_distribution.png',
            'categorical_features_vs_risk.png',
            'purpose_vs_risk.png',
            'correlation_heatmap.png',
            'confusion_matrices_comparison.png',
            'roc_curves_comparison.png',
            'metrics_comparison_bar_chart.png',
            'cv_score_distribution.png',
            'feature_importances.png'
        ]
        
        print("\nVerifying output plots:")
        missing_plots = []
        for plot in expected_plots:
            if os.path.exists(plot):
                print(f"  [x] Found: {plot} (size: {os.path.getsize(plot)} bytes)")
            else:
                print(f"  [ ] Missing: {plot}")
                missing_plots.append(plot)
                
        if missing_plots:
            print(f"\nWarning: Some expected plots were not found: {missing_plots}")
            sys.exit(1)
        else:
            print("\nAll plots verified successfully!")
            
    except Exception as e:
        print("=" * 60)
        print("FAILURE: Execution error encountered:")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    verify()
