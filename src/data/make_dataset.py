from pathlib import Path
import re
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)

def clean_text(s: str) -> str:
    """Light cleaning: collapse whitespace and trim."""
    if not isinstance(s, str):
        return ""
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def read_csv_safely(path: Path) -> pd.DataFrame:
    """Read CSV with sane fallbacks for Windows-encoded files."""
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        # Try common encodings if UTF-8 fails
        for enc in ("utf-8-sig", "cp1252", "latin1"):
            try:
                return pd.read_csv(path, encoding=enc, on_bad_lines="skip")
            except Exception:
                continue
        # Last resort
        return pd.read_csv(path, encoding_errors="ignore", on_bad_lines="skip")

def main():
    fake_path = RAW_DIR / "Fake.csv"
    true_path = RAW_DIR / "True.csv"

    if not fake_path.exists() or not true_path.exists():
        raise FileNotFoundError(
            f"CSV files not found. Expected:\n- {fake_path}\n- {true_path}"
        )

    fake = read_csv_safely(fake_path)
    true = read_csv_safely(true_path)

    # Minimal required columns
    required_cols = {"title", "text"}
    if not required_cols.issubset(fake.columns) or not required_cols.issubset(true.columns):
        raise ValueError(
            f"Both CSVs must contain columns: {sorted(required_cols)}.\n"
            f"Fake columns: {list(fake.columns)}\nTrue columns: {list(true.columns)}"
        )

    # Clean and label
    for df, lbl in ((fake, 0), (true, 1)):
        df["title"] = df["title"].map(clean_text)
        df["text"] = df["text"].map(clean_text)
        df["label"] = lbl

    # Combine & shuffle
    all_df = pd.concat([fake, true], ignore_index=True)
    all_df = all_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # 70/15/15 split with stratification
    train_df, test_df = train_test_split(
        all_df, test_size=0.15, random_state=42, stratify=all_df["label"]
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.1765, random_state=42, stratify=train_df["label"]
    )  # 0.85 * 0.1765 ≈ 0.15

    # Save
    out_train = PROC_DIR / "train.csv"
    out_val = PROC_DIR / "val.csv"
    out_test = PROC_DIR / "test.csv"

    train_df.to_csv(out_train, index=False)
    val_df.to_csv(out_val, index=False)
    test_df.to_csv(out_test, index=False)

    # Helpful prints
    print("Saved processed splits to data/processed/")
    print(f"train.csv: {len(train_df)} rows  | label counts:\n{train_df['label'].value_counts()}")
    print(f"val.csv:   {len(val_df)} rows  | label counts:\n{val_df['label'].value_counts()}")
    print(f"test.csv:  {len(test_df)} rows | label counts:\n{test_df['label'].value_counts()}")

if __name__ == "__main__":
    main()