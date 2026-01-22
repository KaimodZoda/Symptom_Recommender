import pandas as pd
import json
import os
import re

# Translation libraries (install: pip install deep-translator rapidfuzz)
try:
    from deep_translator import GoogleTranslator
    from rapidfuzz import process
    HAS_TRANSLATION = True
except ImportError:
    HAS_TRANSLATION = False
    print("Warning: deep-translator and/or rapidfuzz not installed. Translation disabled.")
    print("Install with: pip install deep-translator rapidfuzz")

def is_english(text):
    """Check if text is primarily English."""
    return bool(re.match(r'^[a-zA-Z\s\-/()]+$', str(text)))

def extract_data(json_string):
    """Extract symptoms from JSON summary."""
    if pd.isna(json_string) or str(json_string).strip() == "":
        return pd.Series([[], [], []])
    try:
        data = json.loads(json_string)
        return pd.Series([
            data.get('no_symptoms', []),
            data.get('idk_symptoms', []),
            data.get('yes_symptoms', [])
        ])
    except:
        return pd.Series([[], [], []])


def extract_symptom_texts(symptom_list):
    """Extract text from symptom list of dicts."""
    if not isinstance(symptom_list, list) or len(symptom_list) == 0:
        return []
    return [item.get('text') for item in symptom_list if isinstance(item, dict) and 'text' in item]


def create_symptom_mapping(file_path, output_file="symptom_mapping.csv"):
    """
    Extract all unique symptoms from yes, no, idk columns and create mapping CSV.
    
    Args:
        file_path: Path to the Excel data file
        output_file: Output CSV filename
    """
    # Load data
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    print("Loading data...")
    df = pd.read_excel(file_path)
    print(f"Loaded {len(df)} records.")
    
    # Extract symptoms from JSON
    print("Extracting symptoms from JSON summary...")
    df[['no_symptoms', 'idk_symptoms', 'yes_symptoms']] = df['summary'].apply(extract_data)
    
    # Collect all unique symptoms
    all_symptoms = set()
    
    for col in ['yes_symptoms', 'no_symptoms', 'idk_symptoms']:
        extracted = df[col].apply(extract_symptom_texts)
        unique_symptoms = extracted.explode().dropna().unique()
        all_symptoms.update(unique_symptoms)
    
    print(f"Found {len(all_symptoms)} unique symptoms.")
    
    # Separate English and Thai symptoms
    english_symptoms = {s.strip() for s in all_symptoms if is_english(s)}
    thai_symptoms = {s.strip() for s in all_symptoms if not is_english(s) and s.strip() != ""}
    
    print(f"  - English: {len(english_symptoms)}")
    print(f"  - Thai: {len(thai_symptoms)}")
    
    # Create mapping dataframe with translation
    mapping_data = []
    
    # Add Thai symptoms (map to themselves)
    for symptom in thai_symptoms:
        mapping_data.append({
            'English_Raw': symptom.lower(),
            'Thai_Standard': symptom
        })
    
    # Translate English symptoms
    if HAS_TRANSLATION and len(english_symptoms) > 0:
        print("\nTranslating English symptoms to Thai...")
        translator = GoogleTranslator(source='en', target='th')
        
        for eng_symptom in sorted(english_symptoms):
            eng_lower = eng_symptom.lower()
            try:
                # Translate to Thai
                translated = translator.translate(eng_lower)
                
                # Fuzzy match with existing Thai symptoms to avoid duplicates
                match = process.extractOne(translated, thai_symptoms, score_cutoff=80)
                
                if match:
                    # Use existing Thai symptom
                    thai_standard = match[0]
                    print(f"  {eng_symptom} → {translated} (matched: {thai_standard})")
                else:
                    # Use new translation
                    thai_standard = translated
                    thai_symptoms.add(translated)
                    print(f"  {eng_symptom} → {thai_standard}")
                
                mapping_data.append({
                    'English_Raw': eng_lower,
                    'Thai_Standard': thai_standard
                })
            except Exception as e:
                # If translation fails, map to original
                print(f"  {eng_symptom} → [Translation failed, using original]")
                mapping_data.append({
                    'English_Raw': eng_lower,
                    'Thai_Standard': eng_symptom
                })
    else:
        # No translation available, map English to themselves
        for eng_symptom in english_symptoms:
            mapping_data.append({
                'English_Raw': eng_symptom.lower(),
                'Thai_Standard': eng_symptom
            })
    
    mapping_df = pd.DataFrame(mapping_data)
    
    # Remove duplicates (keep first occurrence)
    mapping_df = mapping_df.drop_duplicates(subset=['English_Raw'], keep='first')
    
    # Export to CSV
    mapping_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✓ Mapping file created: {output_file}")
    
    # Export to pickle
    pkl_file = output_file.replace('.csv', '.pkl')
    mapping_df.to_pickle(pkl_file)
    print(f"✓ Mapping file created: {pkl_file}")
    print(f"✓ Total mappings: {len(mapping_df)}")
    
    # Display samples - separate Thai and English
    print("\n--- Sample Thai Mappings (First 5) ---")
    thai_mappings = mapping_df[mapping_df['Thai_Standard'].apply(lambda x: not is_english(x))]
    print(thai_mappings.head(5).to_string(index=False))
    
    print("\n--- Sample English → Thai Translations (First 10) ---")
    english_mappings = mapping_df[mapping_df['English_Raw'].apply(lambda x: is_english(x))]
    if len(english_mappings) > 0:
        print(english_mappings.head(10).to_string(index=False))
    else:
        print("No English symptoms found.")
    
    return mapping_df


# --- Main Execution ---
if __name__ == "__main__":
    file_path = "[CONFIDENTIAL] AI symptom picker data (Agnos candidate assignment).xlsx"
    
    try:
        mapping_df = create_symptom_mapping(file_path, output_file="symptom_mapping.csv")
        print("\n✓ Process completed successfully!")
    except Exception as e:
        print(f"\n✗ Error: {e}")