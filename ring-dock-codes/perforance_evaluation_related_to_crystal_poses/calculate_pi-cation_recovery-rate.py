import pandas as pd
import numpy as np

def check_protein_matching(report_file, predictions_file):
    """
    Check how many proteins from the report file have at least one match in the predictions file
    """
    # Load both CSV files[6,7,8](@ref)
    df_report = pd.read_csv(report_file)
    df_pred = pd.read_csv(predictions_file)
    
    # Standardize column names by stripping whitespace
    df_report.columns = df_report.columns.str.strip()
    df_pred.columns = df_pred.columns.str.strip()
    
    # Extract protein residue identifiers from both datasets
    def extract_residue(protein_str):
        """Extract residue information from protein string"""
        if isinstance(protein_str, str) and '-' in protein_str:
            parts = protein_str.split('-')
            if len(parts) >= 2:
                return f"{parts[0]}-{parts[1]}"
        return protein_str
    
    df_report['Protein_Residue'] = df_report['Protein'].apply(extract_residue)
    df_pred['Protein_Residue'] = df_pred['Protein'].apply(extract_residue)
    
    # Get unique proteins from both datasets
    unique_report_proteins = set(df_report['Protein_Residue'].unique())
    unique_pred_proteins = set(df_pred['Protein_Residue'].unique())
    
    # Find proteins that exist in both datasets
    matched_proteins = unique_report_proteins.intersection(unique_pred_proteins)
    
    # Calculate statistics
    total_report_proteins = len(unique_report_proteins)
    matched_count = len(matched_proteins)
    match_percentage = (matched_count / total_report_proteins * 100) if total_report_proteins > 0 else 0
    
    # Create detailed report
    results = {
        'total_proteins_in_report': total_report_proteins,
        'matched_proteins_count': matched_count,
        'match_percentage': match_percentage,
        'unmatched_proteins': list(unique_report_proteins - matched_proteins),
        'matched_proteins_list': list(matched_proteins)
    }
    
    return results

def save_matching_report(results, output_file="protein_matching_report.csv"):
    """
    Save the matching results to a CSV file
    """
    # Create DataFrame for the summary
    summary_data = {
        'Metric': [
            'Total proteins in report',
            'Matched proteins count',
            'Match percentage (%)'
        ],
        'Value': [
            results['total_proteins_in_report'],
            results['matched_proteins_count'],
            f"{results['match_percentage']:.2f}%"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create DataFrame for detailed unmatched proteins
    unmatched_df = pd.DataFrame({
        'Unmatched_Proteins': results['unmatched_proteins']
    })
    
    # Create DataFrame for detailed matched proteins
    matched_df = pd.DataFrame({
        'Matched_Proteins': results['matched_proteins_list']
    })
    
    # Save to Excel with multiple sheets
    with pd.ExcelWriter(output_file.replace('.csv', '.xlsx')) as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        unmatched_df.to_excel(writer, sheet_name='Unmatched_Proteins', index=False)
        matched_df.to_excel(writer, sheet_name='Matched_Proteins', index=False)
    
    print(f"Detailed matching report saved to: {output_file.replace('.csv', '.xlsx')}")

def main():
    report_file = "all_pication_interactions.csv"  #make sure paths are corrected set
    predictions_file = "predictions_with_energy_ranked_new.csv"
    
    try:
        print("Checking protein matching between report and prediction files...")
        results = check_protein_matching(report_file, predictions_file)
        
        # Display summary results
        print(f"\n=== PROTEIN MATCHING RESULTS ===")
        print(f"Total proteins in report file: {results['total_proteins_in_report']}")
        print(f"Proteins matched in prediction file: {results['matched_proteins_count']}")
        print(f"Match percentage: {results['match_percentage']:.2f}%")
        
        print(f"\nNumber of unmatched proteins: {len(results['unmatched_proteins'])}")
        if results['unmatched_proteins']:
            print("Sample of unmatched proteins (first 10):")
            for protein in results['unmatched_proteins'][:10]:
                print(f"  - {protein}")
        
        # Save detailed report
        save_matching_report(results)
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    main()
