import pandas as pd

# Configure pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)
pd.set_option('display.max_colwidth', 50)

print("=" * 80)
print("SYMPTOM MAPPING")
print("=" * 80)
df = pd.read_pickle('symptom_mapping.pkl')
print(df.head(10).to_string(index=False))

print("\n" + "=" * 80)
print("ASSOCIATION RULES")
print("=" * 80)
rules = pd.read_pickle('association_rules.pkl')
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10).to_string(index=True))

print("\n" + "=" * 80)
print("PREPROCESSED DATA")
print("=" * 80)
data = pd.read_csv('preprocessed_data.csv')
print(f"Shape: {data.shape}")
print(f"Columns: {list(data.columns)}")
print("\nFirst 5 records:")
print(data.head(5).to_string(index=True))
print("\n" + "=" * 80)