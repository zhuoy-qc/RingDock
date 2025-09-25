import pandas as pd
import numpy as np

def check_protein_matching(report_file, predictions_file):
    """
    Check how many proteins from the report file have at least one match in the predictions file
    """
    # Load both CSV files
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
        'total_pi_cation_interactions': total_report_proteins,
        'recovered_interactions': matched_count,
        'recovery_rate': match_percentage,
        'missed_interactions': list(unique_report_proteins - matched_proteins),
        'matched_interactions_list': list(matched_proteins)
    }

    return results

def save_matching_report(results, output_file="pi_cation_matching_report.csv"):
    """
    Save the matching results to a CSV file
    """
    # Create DataFrame for the summary
    summary_data = {
        'Metric': [
            'Total PI-CATION interactions in experimental crystal structures',
            'Recovered in docked ring poses',
            'Recovery rate (%)'
        ],
        'Value': [
            results['total_pi_cation_interactions'],
            results['recovered_interactions'],
            f"{results['recovery_rate']:.2f}%"
        ]
    }

    summary_df = pd.DataFrame(summary_data)

    # Create DataFrame for detailed missed interactions
    missed_df = pd.DataFrame({
        'Missed_Interactions': results['missed_interactions']
    })

    # Create DataFrame for detailed matched interactions
    matched_df = pd.DataFrame({
        'Matched_Interactions': results['matched_interactions_list']
    })

    # Save summary to CSV
    summary_df.to_csv(output_file, index=False)
    
    print(f"Summary report saved to: {output_file}")

def main():
    report_file = "/data1/zyin/posebuster_dataset/newest_pb/posebusters_benchmark_set/pication_interactions_report.csv"
    predictions_file = "predictions_with_energy_ranked.csv"

    try:
        print("Running ...")
        results = check_protein_matching(report_file, predictions_file)

        # Display summary results
        print(f"\n=== PI-CATION INTERACTION RECOVERY RATE  ===")
        print(f"Total PI-CATION interactions in experimental crystal structures: {results['total_pi_cation_interactions']}")
        print(f"Recovered in docked ring poses: {results['recovered_interactions']}")
        print(f"Recovery rate: {results['recovery_rate']:.2f}%")

        print(f"\nMissed pi-cation interaction (labelled by protein cation side chain): {len(results['missed_interactions'])}")
        if results['missed_interactions']:
            print("Sample of missed ones (first 10):")
            for interaction in results['missed_interactions'][:10]:
                print(f"  - {interaction}")

        # Save detailed report
        save_matching_report(results)

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    main()
