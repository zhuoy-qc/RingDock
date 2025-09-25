import pandas as pd
import numpy as np

def load_and_preprocess_data(predictions_file, report_file):
    """
    Load and preprocess two CSV files
    """
    df_pred = pd.read_csv(predictions_file)
    df_report = pd.read_csv(report_file)

    # Standardize column names by stripping whitespace
    df_pred.columns = df_pred.columns.str.strip()
    df_report.columns = df_report.columns.str.strip()

    # Rename columns in report file to match prediction file
    column_mapping = {
        'Distance_Å': 'Distance',
        'Offset_Å': 'Offset',
        'RZ_Å': 'RZ',
        'Angle_°': 'Angle'
    }
    # Only rename columns that actually exist in the DataFrame
    existing_columns = {k: v for k, v in column_mapping.items() if k in df_report.columns}
    df_report = df_report.rename(columns=existing_columns)

    return df_pred, df_report

def extract_protein_residue(protein_str):
    """
    Extract residue information from protein string
    Example: "ARG-274-C" -> "ARG-274"
    """
    if isinstance(protein_str, str) and '-' in protein_str:
        parts = protein_str.split('-')
        if len(parts) >= 2:
            return f"{parts[0]}-{parts[1]}"
    return protein_str

def extract_pdb_id(directory_str):
    """
    Extract PDB ID from Directory string
    Example: "complexes_6QU4_JJ2_6QU4_JJ2_protein_pyrrole" -> "6QU4"
    """
    if isinstance(directory_str, str) and '_' in directory_str:
        parts = directory_str.split('_')
        for part in parts:
            if len(part) == 4 and part.isalnum():
                return part
    return directory_str

def calculate_absolute_errors(df_pred, df_report, rank_threshold=5):
    """
    Calculate absolute errors between two DataFrames, filtering by prediction rank
    """
    # Create standardized residue identifiers for merging
    df_pred['Protein_Residue'] = df_pred['Protein'].apply(extract_protein_residue)
    df_report['Protein_Residue'] = df_report['Protein'].apply(extract_protein_residue)
    df_pred['PDB_ID'] = df_pred['Directory'].apply(extract_pdb_id)

    # Filter by interaction type if the column exists
    if 'Interaction_Type' in df_pred.columns and 'Interaction_Type' in df_report.columns:
        df_pred = df_pred[df_pred['Interaction_Type'] == 'π-Cation']
        df_report = df_report[df_report['Interaction_Type'] == 'π-Cation']

    # Filter by rank threshold
    df_pred = df_pred[df_pred['Energy_Rank'] <= rank_threshold]

    # Merge on Protein_Residue
    merged_df = pd.merge(
        df_pred,
        df_report,
        on=['Protein_Residue'],
        suffixes=('_pred', '_report'),
        how='inner'
    )

    if merged_df.empty:
        print("Warning: No matching protein residues found")
        return pd.DataFrame()

    # Calculate absolute errors
    results = []
    for _, row in merged_df.iterrows():
        distance_error = abs(row['Distance_pred'] - row['Distance_report'])
        offset_error = abs(row['Offset_pred'] - row['Offset_report'])
        rz_error = abs(row['RZ_pred'] - row['RZ_report'])

        results.append({
            'PDB_ID': row['PDB_ID'],
            'Protein_Residue': row['Protein_Residue'],
            'Distance_Error': distance_error,
            'Offset_Error': offset_error,
            'RZ_Error': rz_error,
            'Pred_Rank': row['Energy_Rank'],
            'Pred_Energy': row['Predicted_Energy']
        })

    # Create DataFrame and round error columns to 2 decimal places
    error_df = pd.DataFrame(results)
    # Round the error columns to 2 decimal places[1,2,5](@ref)
    error_cols_to_round = ['Distance_Error', 'Offset_Error', 'RZ_Error']
    error_df[error_cols_to_round] = error_df[error_cols_to_round].round(2)
    
    return error_df

def main():
    predictions_file = "predictions_with_energy_ranked.csv"
    report_file = "pication_interactions_report.csv"
    rank_threshold = 5  # Only include predictions ranked 1 to 5

    try:
        df_pred, df_report = load_and_preprocess_data(predictions_file, report_file)
        error_df = calculate_absolute_errors(df_pred, df_report, rank_threshold)

        if not error_df.empty:
            # Select only the error columns for statistics
            error_columns_for_stats = ['Distance_Error', 'Offset_Error', 'RZ_Error']
            
            # Calculate required statistics for display
            mean_values = error_df[error_columns_for_stats].mean()
            median_values = error_df[error_columns_for_stats].median()
            max_values = error_df[error_columns_for_stats].max()
            
            # Create summary DataFrame for display
            summary_df = pd.DataFrame({
                'mean': mean_values,
                'median': median_values,
                'max': max_values
            }).T  # Transpose to have statistics as rows and metrics as columns
            
            # Round the summary statistics to 2 decimal places for cleaner display[1,2,5](@ref)
            summary_df = summary_df.round(2)
            
            print("Error Statistics Summary:")
            # Use to_string with float_format to ensure 2 decimal places in output[6](@ref)
            print(summary_df.to_string(float_format='%.2f'))
            
            # Display top 20 largest Distance_Errors
            print("\nTop 20 Largest Distance_Errors:")
            # Sort by Distance_Error in descending order and select top 20
            top20_distance_errors = error_df.nlargest(20, 'Distance_Error')[['PDB_ID', 'Protein_Residue', 'Distance_Error']]
            # Ensure Distance_Error is displayed with 2 decimal places[6](@ref)
            print(top20_distance_errors.to_string(index=False, float_format='%.2f'))
            
            # Save complete results to CSV (including all columns)
            output_columns = [
                'PDB_ID',
                'Protein_Residue',
                'Distance_Error',
                'Offset_Error',
                'RZ_Error',
                'Pred_Rank',
                'Pred_Energy'
            ]
            # Round the error columns in the DataFrame before saving to CSV for consistency[1,2,5](@ref)
            error_df_to_save = error_df[output_columns].copy()
            error_cols = ['Distance_Error', 'Offset_Error', 'RZ_Error']
            error_df_to_save[error_cols] = error_df_to_save[error_cols].round(2)
            error_df_to_save.to_csv("absolute_errors_report_new.csv", index=False)
            print("\nComplete results saved to: absolute_errors_report.csv")
        else:
            print("No matching data found for calculation")

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    main()
