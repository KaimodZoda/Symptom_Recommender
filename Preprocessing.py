import pandas as pd
import json
import os

# --- 1. Symptom Standardizer ---
class SymptomStandardizer:
    def __init__(self, mapping_file="symptom_mapping.pkl"):
        self.master_map = {}
        self.mapping_file = mapping_file
        self.load_mapping()

    def load_mapping(self):
        if os.path.exists(self.mapping_file):
            try:
                mapping_df = pd.read_pickle(self.mapping_file)
                self.master_map = dict(zip(mapping_df['English_Raw'].str.lower(), mapping_df['Thai_Standard']))
                print(f"Loaded {len(self.master_map)} mapping pairs.")
            except Exception as e:
                print(f"Error loading mapping file: {e}")
        else:
            print(f"Warning: {self.mapping_file} not found.")

    def get_standard(self, text):
        if not isinstance(text, str):
            return text
        clean = text.lower().strip()
        return self.master_map.get(clean, clean)


# --- 2. Data Processing Utilities ---
def extract_data(json_string):
    """Extract diseases, procedures, and symptoms from JSON summary."""
    if pd.isna(json_string) or str(json_string).strip() == "":
        return pd.Series([[], [], [], [], []])
    try:
        data = json.loads(json_string)
        return pd.Series([
            data.get('diseases', []), 
            data.get('procedures', []),
            data.get('no_symptoms', []), 
            data.get('idk_symptoms', []),
            data.get('yes_symptoms', [])
        ])
    except:
        return pd.Series([[], [], [], [], []])


def process_list(s_list, standardizer, is_yes=False):
    """Standardize symptom names from a list of dicts."""
    if not isinstance(s_list, list) or len(s_list) == 0:
        return []
    
    items = s_list.copy()
    if is_yes and len(items) > 0:
        items.pop()  # Remove the last (ground truth) item for training
    
    return [standardizer.get_standard(item.get('text')) for item in items if 'text' in item]


def prepare_basket(row):
    """Create a transaction basket for Apriori mining."""
    items = [f"G_{row['gender_code']}", str(row['age_group'])]
    items.extend([f"Yes_{s}" for s in row['yes_list']])
    return items


# --- 3. Data Loading and Preprocessing ---
def load_and_preprocess_data(file_path, mapping_file="symptom_mapping.pkl"):
    """
    Load Excel data, standardize symptoms, and encode demographics.
    
    Args:
        file_path: Path to the Excel data file.
        mapping_file: Path to the symptom mapping pickle file.
    
    Returns:
        df: Full dataset with processed columns.
        standardizer: Initialized SymptomStandardizer instance.
    """
    # Load data
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    df = pd.read_excel(file_path)
    print("File loaded.")
    
    # Initialize standardizer
    standardizer = SymptomStandardizer(mapping_file)
    
    # Extract structured data from JSON summary
    df[['diseases', 'procedures', 'no_symptoms', 'idk_symptoms', 'yes_symptoms']] = df['summary'].apply(extract_data)
    
    # Process symptom lists
    df['yes_list'] = df['yes_symptoms'].apply(lambda x: process_list(x, standardizer, is_yes=True))
    df['no_list'] = df['no_symptoms'].apply(lambda x: process_list(x, standardizer))
    df['idk_list'] = df['idk_symptoms'].apply(lambda x: process_list(x, standardizer))
    
    # Encode demographics
    df['gender_code'] = df['gender'].map({'male': 'M', 'female': 'F'})
    df['age_group'] = pd.cut(df['age'], bins=[0, 45, 60, 100], labels=['Age_Adult', 'Age_Middle', 'Age_Senior'])
    
    # Drop columns with only empty lists
    for col in ['yes_list', 'no_list', 'idk_list', 'diseases', 'procedures']:
        if col in df.columns and df[col].apply(lambda x: isinstance(x, list) and len(x) == 0).all():
            df.drop(columns=[col], inplace=True)
            print(f"Dropped column '{col}' (all empty lists)")
    
    # Remove raw symptom columns and summary after extraction
    df.drop(columns=['summary', 'yes_symptoms', 'no_symptoms', 'idk_symptoms'], inplace=True, errors='ignore')
    
    print(f"Total records: {len(df)}")
    
    return df, standardizer


# --- 4. Main Execution ---
if __name__ == "__main__":
    file_path = "[CONFIDENTIAL] AI symptom picker data (Agnos candidate assignment).xlsx"
    
    # Load and preprocess
    df, standardizer = load_and_preprocess_data(file_path)
    
    # Create baskets
    df['basket'] = df.apply(prepare_basket, axis=1)
    print("\nBasket creation complete.")
    
    # Display sample
    print("\n--- Sample Records (First 10 rows) ---")
    print(df.head(10))
    
    print(f"\n--- Final DataFrame Info ---")
    print(f"Shape: {df.shape}")
    
    # Export to CSV
    output_file = "preprocessed_data.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✓ Data exported to {output_file}")