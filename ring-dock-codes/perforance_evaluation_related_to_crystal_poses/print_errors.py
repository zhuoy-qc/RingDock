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

def extract_protein_residue_with_chain(protein_str):
    """
    Extract residue information from protein string including chain
    Example: "ARG-274-C" -> "ARG-274-C"
    """
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
    # Create standardized residue identifiers for merging (including chain)
    df_pred['Protein_Residue_Chain'] = df_pred['Protein'].apply(extract_protein_residue_with_chain)
    df_report['Protein_Residue_Chain'] = df_report['Protein'].apply(extract_protein_residue_with_chain)
    df_pred['PDB_ID'] = df_pred['Directory'].apply(extract_pdb_id)

    # Filter by interaction type if the column exists
    if 'Interaction_Type' in df_pred.columns and 'Interaction_Type' in df_report.columns:
        df_pred = df_pred[df_pred['Interaction_Type'] == 'π-Cation']
        df_report = df_report[df_report['Interaction_Type'] == 'π-Cation']

    # Filter by rank threshold
    df_pred = df_pred[df_pred['Energy_Rank'] <= rank_threshold]

    # Merge on Protein_Residue_Chain (including chain information)
    merged_df = pd.merge(
        df_pred,
        df_report,
        on=['Protein_Residue_Chain'],  # Changed from 'Protein_Residue' to 'Protein_Residue_Chain'
        suffixes=('_pred', '_report'),
        how='inner'
    )

    if merged_df.empty:
        print("Warning: No matching protein residues found after including chain information")
        return pd.DataFrame()

    # Calculate absolute errors
    results = []
    for _, row in merged_df.iterrows():
        distance_error = abs(row['Distance_pred'] - row['Distance_report'])
        offset_error = abs(row['Offset_pred'] - row['Offset_report'])
        rz_error = abs(row['RZ_pred'] - row['RZ_report'])

        results.append({
            'PDB_ID': row['PDB_ID'],
            'Protein_Cation': row['Protein_Residue_Chain'],  # Renamed for display
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

def calculate_top_errors(error_df, top_n):
    """
    Calculate error statistics for top N ranked predictions
    """
    if error_df.empty:
        return pd.DataFrame()
    
    # Filter for top N ranks
    top_n_df = error_df[error_df['Pred_Rank'] <= top_n]
    
    if top_n_df.empty:
        return pd.DataFrame()
    
    # Calculate statistics for the top N data
    error_columns_for_stats = ['Distance_Error', 'Offset_Error', 'RZ_Error']
    
    # Calculate required statistics for display
    mean_values = top_n_df[error_columns_for_stats].mean()
    median_values = top_n_df[error_columns_for_stats].median()
    max_values = top_n_df[error_columns_for_stats].max()
    
    # Create summary DataFrame for display
    summary_df = pd.DataFrame({
        'mean': mean_values,
        'median': median_values,
        'max': max_values,
    }).T  # Transpose to have statistics as rows and metrics as columns
    
    # Round the summary statistics to 2 decimal places for cleaner display[1,2,5](@ref)
    summary_df = summary_df.round(2)
    
    return summary_df, top_n_df

def main():
    predictions_file = "predictions_with_energy_ranked.csv"
    report_file = "reference_experimental_pication_interactions_report.csv"
    rank_threshold = 5  # Only include predictions ranked 1 to 5

    try:
        df_pred, df_report = load_and_preprocess_data(predictions_file, report_file)
        error_df = calculate_absolute_errors(df_pred, df_report, rank_threshold)

        if not error_df.empty:
            # Calculate and print statistics for top 1
            print("=== TOP 1 ERROR STATISTICS ===")
            top1_summary, top1_df = calculate_top_errors(error_df, 1)
            if not top1_summary.empty:
                print("Error Statistics Summary (Top 1):")
                print(top1_summary.to_string(float_format='%.2f'))
            else:
                print("No top 1 predictions found")
            
            print("\n" + "="*50)
            
            # Calculate and print statistics for top 5
            print("=== TOP 5 ERROR STATISTICS ===")
            top5_summary, top5_df = calculate_top_errors(error_df, 5)
            if not top5_summary.empty:
                print("Error Statistics Summary (Top 5):")
                print(top5_summary.to_string(float_format='%.2f'))
            else:
                print("No top 5 predictions found")

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    main()
