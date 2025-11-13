import argparse
import ast
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

def load_pipeline(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    obj = joblib.load(path)
    if not all(k in obj for k in ("model", "scaler", "feature_columns")):
        raise ValueError("Loaded joblib does not contain required keys: 'model','scaler','feature_columns'")
    return obj["model"], obj["scaler"], obj["feature_columns"]

def prepare_dataframe(df: pd.DataFrame, feature_columns):
    """
    Align dataframe to feature_columns expected by model.
    - Add missing cols with 0
    - Drop extra columns
    - Reorder to the exact column order
    """
    df_proc = df.copy()
    # Ensure numeric dtype for all columns we will use
    for c in df_proc.columns:
        # try convert to numeric where possible
        try:
            df_proc[c] = pd.to_numeric(df_proc[c])
        except Exception:
            pass

    # Add missing columns with zeros
    missing = [c for c in feature_columns if c not in df_proc.columns]
    if missing:
        print(f"Warning: {len(missing)} missing columns will be filled with 0. Missing examples: {missing[:8]}")
        for c in missing:
            df_proc[c] = 0

    # Keep only the feature_columns (drop extras)
    extra = [c for c in df_proc.columns if c not in feature_columns]
    if extra:
        print(f"Note: the input contains {len(extra)} extra columns that will be ignored (examples: {extra[:8]})")

    df_proc = df_proc.reindex(columns=feature_columns, fill_value=0)
    return df_proc

def predict_df(model, scaler, feature_columns, df_input):
    df_prepared = prepare_dataframe(df_input, feature_columns)
    X = df_prepared.values.astype(float)
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:, 1]
    preds = model.predict(X_scaled)
    out = df_input.copy().reset_index(drop=True)
    out["churn_probability"] = probs
    out["churn_pred"] = preds
    return out

def parse_kv_input(s: str):
    """
    Parse interactive user input that can be:
      - a JSON object like: {"tenure": 12, "MonthlyCharges": 75.3}
      - a Python dict literal: {'tenure':12, 'MonthlyCharges':75}
      - key=value pairs separated by commas: tenure=12,MonthlyCharges=75
    Returns a dict of parsed values (keys as strings).
    """
    s = s.strip()
    if not s:
        return {}
    # Try JSON first
    try:
        data = json.loads(s)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    # Try python literal via ast
    try:
        data = ast.literal_eval(s)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    # Fallback: parse key=value pairs
    out = {}
    for part in s.split(","):
        if not part.strip():
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            k = k.strip()
            v = v.strip()
            # try convert numeric
            try:
                vv = float(v) if "." in v else int(v)
            except Exception:
                vv = v
            out[k] = vv
        else:
            # single token - ignore
            pass
    return out

def interactive_predict(model, scaler, feature_columns):
    print("\nInteractive mode — supply customer features as JSON/dict or key=value pairs.")
    print("Example JSON: {\"tenure\": 12, \"MonthlyCharges\": 70.0, \"TotalCharges\": 840.0}")
    print("Example kv : tenure=12,MonthlyCharges=70,TotalCharges=840")
    print("\nAvailable target feature columns (first 60 shown):")
    for i, c in enumerate(feature_columns[:60], 1):
        print(f"{i:02d}. {c}")
    print("... (total columns: {})\n".format(len(feature_columns)))

    raw = input("Paste input now (or press Enter to quit):\n> ").strip()
    if not raw:
        print("No input provided. Exiting.")
        return

    parsed = parse_kv_input(raw)
    if not parsed:
        print("Could not parse input or empty dict. Exiting.")
        return

    # Build single-row dataframe
    df = pd.DataFrame([parsed])
    out = predict_df(model, scaler, feature_columns, df)
    row = out.iloc[0]
    print("\nPrediction result:")
    print(f"  churn probability: {row['churn_probability']:.6f}")
    print(f"  predicted class   : {int(row['churn_pred'])}   (1=churn, 0=no churn)")
    # Provide a simple suggested action based on threshold 0.5
    thr = 0.5
    if row["churn_probability"] >= thr:
        print(f"  > Model suggests this customer is HIGH RISK (prob >= {thr})")
    else:
        print(f"  > Model suggests this customer is LOW RISK (prob < {thr})")

def batch_predict(model, scaler, feature_columns, csv_path: Path, out_csv: Path = None):
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    results = predict_df(model, scaler, feature_columns, df)
    if out_csv:
        results.to_csv(out_csv, index=False)
        print(f"Saved predictions to {out_csv}")
    else:
        print("Predictions:")
        print(results[["churn_probability", "churn_pred"]].head(20).to_string(index=False))

def main():
    p = argparse.ArgumentParser(description="Load churn_model.joblib and predict churn for input customers.")
    p.add_argument("--model", type=str, default="churn_model.joblib", help="path to joblib model (default: churn_model.joblib)")
    p.add_argument("--csv", type=str, help="CSV file of customers to predict (batch).")
    p.add_argument("--out", type=str, help="output CSV file path for batch predictions.")
    p.add_argument("--interactive", action="store_true", help="start interactive prompt for single prediction.")
    args = p.parse_args()

    model_path = Path(args.model)
    model, scaler, feature_columns = load_pipeline(model_path)
    print(f"Loaded model from {model_path}. Expecting {len(feature_columns)} feature columns.\n")

    if args.interactive:
        interactive_predict(model, scaler, feature_columns)
        return

    if args.csv:
        csv_path = Path(args.csv)
        out_path = Path(args.out) if args.out else None
        batch_predict(model, scaler, feature_columns, csv_path, out_path)
        return

    print("No mode selected. Use --interactive or --csv <path>. See --help for details.")

if __name__ == "__main__":
    main()
