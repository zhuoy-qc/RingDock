import pandas as pd
import re

# ===== CONFIGURABLE PARAMETERS =====
TOP_PERCENT_PER_RESIDUE = {
    'ARG': 30,   # top % of ARG predictions
    'HIS': 50,   # top % of HIS predictions
    'LYS': 5     # top % of LYS predictions
}
# ===================================

def extract_residue_id(protein_str):
    if pd.isna(protein_str):
        return None
    s = str(protein_str).strip()
    match = re.search(r'([A-Za-z]{3})\W*(\d+)', s)
    if match:
        res_code = match.group(1).upper()
        number = match.group(2)
        if res_code in {'ARG', 'HIS', 'LYS'}:
            return f"{res_code}{number}"
    return None

def extract_residue_type(protein_str):
    """Extract just the residue type: 'ARG', 'HIS', or 'LYS'."""
    rid = extract_residue_id(protein_str)
    if rid:
        return rid[:3]
    return None

def apply_ranking(df_pred, per_residue_pct):
    """
    Filter predictions based on per-residue ranking.
    Returns filtered DataFrame.
    """
    df = df_pred.copy()
    df['ResidueType'] = df['Protein'].apply(extract_residue_type)

    filtered_rows = []
    for res_type in ['ARG', 'HIS', 'LYS']:
        subset = df[df['ResidueType'] == res_type].copy()
        if subset.empty:
            continue
        pct = per_residue_pct.get(res_type, 100)
        n_total = len(subset)
        n_keep = max(1, int(n_total * (pct / 100)))  # keep at least 1 if exists
        subset_sorted = subset.sort_values('Energy_Rank')
        filtered_rows.append(subset_sorted.head(n_keep))
    if filtered_rows:
        return pd.concat(filtered_rows, ignore_index=True)
    else:
        return df.iloc[0:0]  # empty df with same columns

def check_protein_matching_per_residue(report_file, predictions_file, per_residue_pct):
    df_report = pd.read_csv(report_file)
    df_pred = pd.read_csv(predictions_file)

    df_report.columns = df_report.columns.str.strip()
    df_pred.columns = df_pred.columns.str.strip()

    # Apply ranking filter
    df_pred_filtered = apply_ranking(df_pred, per_residue_pct)

    def extract_full_protein_info(protein_str):
        return str(protein_str).strip() if pd.notna(protein_str) else ""

    df_report['Protein_Residue'] = df_report['Protein'].apply(extract_full_protein_info)
    df_pred['Protein_Residue'] = df_pred['Protein'].apply(extract_full_protein_info)
    df_pred_filtered['Protein_Residue'] = df_pred_filtered['Protein'].apply(extract_full_protein_info)

    # Add residue type for both datasets
    df_report['ResidueType'] = df_report['Protein'].apply(extract_residue_type)
    df_pred['ResidueType'] = df_pred['Protein'].apply(extract_residue_type)
    df_pred_filtered['ResidueType'] = df_pred_filtered['Protein'].apply(extract_residue_type)

    # Overall results
    unique_report_all = set(df_report['Protein_Residue'].unique())
    unique_pred_all = set(df_pred['Protein_Residue'].unique())
    unique_pred_filtered_all = set(df_pred_filtered['Protein_Residue'].unique())
    matched_all = unique_report_all & unique_pred_filtered_all

    total_report_all = len(unique_report_all)
    matched_count_all = len(matched_all)
    recovery_rate_all = (matched_count_all / total_report_all * 100) if total_report_all > 0 else 0

    # Count unique residues (by type + number)
    report_res_ids = df_report['Protein'].apply(extract_residue_id).dropna().unique()
    pred_res_ids = df_pred['Protein'].apply(extract_residue_id).dropna().unique()
    pred_filtered_res_ids = df_pred_filtered['Protein'].apply(extract_residue_id).dropna().unique()

    def count_by_type(res_ids):
        counts = {'ARG': 0, 'HIS': 0, 'LYS': 0}
        for rid in res_ids:
            if rid.startswith('ARG'):
                counts['ARG'] += 1
            elif rid.startswith('HIS'):
                counts['HIS'] += 1
            elif rid.startswith('LYS'):
                counts['LYS'] += 1
        return counts

    # Per-residue analysis BEFORE ranking
    per_residue_results_before = {}
    for res_type in ['ARG', 'HIS', 'LYS']:
        report_subset = df_report[df_report['ResidueType'] == res_type]
        pred_subset = df_pred[df_pred['ResidueType'] == res_type]
        
        unique_report_res = set(report_subset['Protein_Residue'].unique()) if not report_subset.empty else set()
        unique_pred_res = set(pred_subset['Protein_Residue'].unique()) if not pred_subset.empty else set()
        matched_res_before = unique_report_res & unique_pred_res

        total_report_res = len(unique_report_res)
        matched_count_res_before = len(matched_res_before)
        recovery_rate_res_before = (matched_count_res_before / total_report_res * 100) if total_report_res > 0 else 0

        per_residue_results_before[res_type] = {
            'total_experimental': total_report_res,
            'recovered_before_ranking': matched_count_res_before,
            'recovery_rate_before_ranking': recovery_rate_res_before
        }

    # Per-residue analysis AFTER ranking
    per_residue_results = {}
    for res_type in ['ARG', 'HIS', 'LYS']:
        report_subset = df_report[df_report['ResidueType'] == res_type]
        pred_filtered_subset = df_pred_filtered[df_pred_filtered['ResidueType'] == res_type]
        
        unique_report_res = set(report_subset['Protein_Residue'].unique()) if not report_subset.empty else set()
        unique_pred_filtered_res = set(pred_filtered_subset['Protein_Residue'].unique()) if not pred_filtered_subset.empty else set()
        matched_res = unique_report_res & unique_pred_filtered_res

        total_report_res = len(unique_report_res)
        matched_count_res = len(matched_res)
        recovery_rate_res = (matched_count_res / total_report_res * 100) if total_report_res > 0 else 0

        per_residue_results[res_type] = {
            'total_experimental': total_report_res,
            'recovered': matched_count_res,
            'recovery_rate': recovery_rate_res,
            'missed_interactions': list(unique_report_res - matched_res),
            'matched_interactions_list': list(matched_res),
            'predictions_considered': len(pred_filtered_subset)
        }

    # Calculate unique predictions NOT in experimental report (BEFORE ranking)
    predictions_not_in_exp_before = unique_pred_all - unique_report_all
    predictions_not_in_exp_before_by_residue = {}
    for res_type in ['ARG', 'HIS', 'LYS']:
        subset_before = df_pred[df_pred['ResidueType'] == res_type]['Protein_Residue']
        subset_unique_before = set(subset_before.unique())
        predictions_not_in_exp_before_by_residue[res_type] = len(subset_unique_before & predictions_not_in_exp_before)

    # Calculate unique predictions NOT in experimental report (AFTER ranking)
    predictions_not_in_exp_after = unique_pred_filtered_all - unique_report_all
    predictions_not_in_exp_after_by_residue = {}
    for res_type in ['ARG', 'HIS', 'LYS']:
        subset_after = df_pred_filtered[df_pred_filtered['ResidueType'] == res_type]['Protein_Residue']
        subset_unique_after = set(subset_after.unique())
        predictions_not_in_exp_after_by_residue[res_type] = len(subset_unique_after & predictions_not_in_exp_after)

    # Calculate percentage of non-experimental predictions filtered out PER RESIDUE (ARG and LYS)
    filtered_out_pct_by_residue = {}
    for res_type in ['ARG', 'LYS']:  # Only ARG and LYS as requested
        before_count = predictions_not_in_exp_before_by_residue.get(res_type, 0)
        after_count = predictions_not_in_exp_after_by_residue.get(res_type, 0)
        if before_count > 0:
            filtered_out = before_count - after_count
            pct = (filtered_out / before_count) * 100
        else:
            pct = 0.0
        filtered_out_pct_by_residue[res_type] = pct

    # Calculate overall recovery before ranking
    matched_all_before = unique_report_all & unique_pred_all
    total_recovered_before = len(matched_all_before)
    overall_rate_before = (total_recovered_before / total_report_all * 100) if total_report_all > 0 else 0

    return {
        'overall': {
            'total_pi_cation_interactions': total_report_all,
            'recovered_interactions': matched_count_all,
            'recovery_rate': recovery_rate_all,
            'missed_interactions': list(unique_report_all - matched_all),
            'matched_interactions_list': list(matched_all),
            'total_predictions_considered': len(df_pred_filtered),
        },
        'overall_before_ranking': {
            'recovered_interactions_before': total_recovered_before,
            'recovery_rate_before': overall_rate_before
        },
        'per_residue_results_before_ranking': per_residue_results_before,
        'per_residue_results': per_residue_results,
        'predictions_not_in_exp_before_ranking': {
            'count': len(predictions_not_in_exp_before),
            'by_residue': predictions_not_in_exp_before_by_residue
        },
        'predictions_not_in_exp_after_ranking': {
            'count': len(predictions_not_in_exp_after),
            'by_residue': predictions_not_in_exp_after_by_residue
        },
        'filtered_out_pct_by_residue': filtered_out_pct_by_residue,  # Updated key
        'per_residue_percent': per_residue_pct,
        'report_residue_counts': count_by_type(report_res_ids),
        'prediction_residue_counts': count_by_type(pred_res_ids),
        'prediction_filtered_residue_counts': count_by_type(pred_filtered_res_ids)
    }

def save_matching_report_txt(results, output_file="pi_cation_matching_report_per_residue.txt"):
    pct = results['per_residue_percent']
    title = f"PER-RESIDUE RANKING (ARG:{pct['ARG']}%, HIS:{pct['HIS']}%, LYS:{pct['LYS']}%)"

    with open(output_file, 'w') as f:
        f.write(f"PI-CATION INTERACTION RECOVERY ANALYSIS ({title})\n")
        f.write("=" * 70 + "\n\n")

        # Before ranking results
        f.write("SAMPLING RECOVERY RATES BEFORE RANKING:\n")
        f.write("-" * 40 + "\n")
        for res_type in ['ARG', 'HIS', 'LYS']:
            res_data = results['per_residue_results_before_ranking'][res_type]
            f.write(f"{res_type}:\n")
            f.write(f"  Total experimental: {res_data['total_experimental']}\n")
            f.write(f"  Recovered: {res_data['recovered_before_ranking']}\n")
            f.write(f"  Recovery rate: {res_data['recovery_rate_before_ranking']:.2f}%\n\n")

        # Overall results
        f.write("OVERALL SUMMARY STATISTICS AFTER RANKING:\n")
        f.write("-" * 40 + "\n")
        overall = results['overall']
        overall_before = results['overall_before_ranking']
        f.write(f"Total experimental π-cation interactions: {overall['total_pi_cation_interactions']}\n")
        f.write(f"Before ranking: {overall_before['recovered_interactions_before']}/{overall['total_pi_cation_interactions']} ({overall_before['recovery_rate_before']:.2f}%)\n")
        f.write(f"After ranking: {overall['recovered_interactions']}/{overall['total_pi_cation_interactions']} ({overall['recovery_rate']:.2f}%)\n")
        f.write(f"Poses considered: {overall['total_predictions_considered']}\n\n")

        # Per-residue results after ranking
        f.write("Each type of protein cation BREAKDOWN AFTER RANKING:\n")
        f.write("-" * 40 + "\n")
        for res_type in ['ARG', 'HIS', 'LYS']:
            res_data = results['per_residue_results'][res_type]
            f.write(f"\n{res_type}:\n")
            f.write(f"  Total experimental: {res_data['total_experimental']}\n")
            f.write(f"  Recovered: {res_data['recovered']}\n")
            f.write(f"  Recovery rate: {res_data['recovery_rate']:.2f}%\n")
            
            if res_data['missed_interactions']:
                f.write(f"  Missed interactions ({len(res_data['missed_interactions'])}):\n")
                for i, inter in enumerate(res_data['missed_interactions'], 1):  # Show ALL interactions
                    f.write(f"    {i}. {inter}\n")
            else:
                f.write("  ✅ All interactions recovered!\n")

        # Filtering efficiency for non-experimental predictions (ARG and LYS)
        f.write("\nFILTERING EFFICIENCY ON NON-EXPERIMENTAL PREDICTIONS:\n")
        f.write("-" * 50 + "\n")
        f.write(f"  ARG: {results['filtered_out_pct_by_residue']['ARG']:.2f}% filtered out\n")
        f.write(f"  LYS: {results['filtered_out_pct_by_residue']['LYS']:.2f}% filtered out\n\n")

        # Residue counts
        f.write("UNIQUE CATIONIC RESIDUE COUNTS (by type):\n")
        f.write("-" * 40 + "\n")
        rpt = results['report_residue_counts']
        pred = results['prediction_residue_counts']
        pred_filtered = results['prediction_filtered_residue_counts']
        f.write("In Experimental Report:\n")
        f.write(f"  ARG: {rpt['ARG']}\n  HIS: {rpt['HIS']}\n  LYS: {rpt['LYS']}\n  Total: {sum(rpt.values())}\n\n")
        f.write("In Prediction Dataset (all poses):\n")
        f.write(f"  ARG: {pred['ARG']}\n  HIS: {pred['HIS']}\n  LYS: {pred['LYS']}\n  Total: {sum(pred.values())}\n\n")
        f.write("In Prediction Dataset (filtered poses):\n")
        f.write(f"  ARG: {pred_filtered['ARG']}\n  HIS: {pred_filtered['HIS']}\n  LYS: {pred_filtered['LYS']}\n  Total: {sum(pred_filtered.values())}\n\n")

        # All missed interactions
        f.write("ALL MISSED INTERACTIONS:\n")
        f.write("-" * 40 + "\n")
        all_missed = overall['missed_interactions']
        if all_missed:
            for i, inter in enumerate(all_missed, 1):
                f.write(f"{i:4d}. {inter}\n")
        else:
            f.write("All interactions recovered!\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("ALL MATCHED INTERACTIONS:\n")
        f.write("-" * 40 + "\n")
        all_matched = overall['matched_interactions_list']
        if all_matched:
            for i, inter in enumerate(all_matched, 1):
                f.write(f"{i:4d}. {inter}\n")
        else:
            f.write("None.\n")

    print(f"Report saved to: {output_file}")

def main():
    report_file = "reference_experimental_pication_interactions_report.csv"
    predictions_file = "predictions_with_energy_ranked.csv"

    try:
        results = check_protein_matching_per_residue(
            report_file, predictions_file,
            per_residue_pct=TOP_PERCENT_PER_RESIDUE
        )

        # Before ranking results
        print(f"\nSAMPLING RECOVERY RATES BEFORE RANKING:")
        print("-" * 60)
        for res_type in ['ARG', 'HIS', 'LYS']:
            res_data = results['per_residue_results_before_ranking'][res_type]
            print(f"{res_type}: {res_data['recovered_before_ranking']}/{res_data['total_experimental']} ({res_data['recovery_rate_before_ranking']:.2f}%)")

        # Overall results before and after ranking
        overall_before = results['overall_before_ranking']
        overall_after = results['overall']
        print(f"\nOVERALL BEFORE RANKING: {overall_before['recovered_interactions_before']}/{overall_after['total_pi_cation_interactions']} ({overall_before['recovery_rate_before']:.2f}%)")
        
        # Each type of protein cation breakdown
        print(f"\nEach type of protein cation BREAKDOWN AFTER RANKING:")
        print("-" * 60)
        total_recovered = 0
        total_experimental = 0
        for res_type in ['ARG', 'HIS', 'LYS']:
            res_data = results['per_residue_results'][res_type]
            recovered = res_data['recovered']
            total_exp = res_data['total_experimental']
            print(f"{res_type}: {recovered}/{total_exp} ({res_data['recovery_rate']:.2f}%)")
            total_recovered += recovered
            total_experimental += total_exp
        
        pi_cation_recovery_rate = (total_recovered / total_experimental * 100) if total_experimental > 0 else 0
        print(f"\npi-cation recovery rate: {total_recovered}/{total_experimental} ({pi_cation_recovery_rate:.2f}%)")

        # Percentage of non-experimental predictions filtered out (ARG and LYS only)
        print(f"\nPERCENTAGE OF NON-EXPERIMENTAL PREDICTIONS FILTERED OUT:")
        print(f"  ARG: {results['filtered_out_pct_by_residue']['ARG']:.2f}%")
        print(f"  LYS: {results['filtered_out_pct_by_residue']['LYS']:.2f}%")

        # Per-residue missed details - PRINT ALL
        print(f"\nPER-RESIDUE MISSED DETAILS AFTER RANKING:")
        print("-" * 60)
        for res_type in ['ARG', 'HIS', 'LYS']:
            res_data = results['per_residue_results'][res_type]
            if res_data['missed_interactions']:
                print(f"\n{res_type} - Missed ({len(res_data['missed_interactions'])} total):")
                for i, m in enumerate(res_data['missed_interactions'], 1):  # Show ALL interactions
                    print(f"  {i}. {m}")
            else:
                print(f"\n{res_type} - ✅ All interactions recovered!")

        save_matching_report_txt(results)

    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
    except Exception as e:
        print(f"💥 Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
