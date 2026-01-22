import pandas as pd
import os
import pickle
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# --- Main Flow: Load Data and Mine Rules ---
preprocessed_file = "preprocessed_data.csv"

if os.path.exists(preprocessed_file):
    print(f"Loading preprocessed data from {preprocessed_file}...")
    df = pd.read_csv(preprocessed_file)
    
    # Parse basket column from string representation
    import ast
    df['basket'] = df['basket'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) and isinstance(x, str) else [])
    
    print(f"Data loaded: {len(df)} records")
    
    # Mine association rules from all data
    print("\nMining association rules...")
    te = TransactionEncoder()
    te_ary = te.fit(df['basket']).transform(df['basket'])
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    frequent_itemsets = apriori(df_encoded, min_support=0.003, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
    
    print(f"Rules mined: {len(rules)}")
    
    # Export rules to pickle
    output_file = "association_rules.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(rules, f)
    
    print(f"✓ Rules exported to {output_file}")
    
    # Display sample rules
    print("\n--- Sample Rules (Top 10 by Lift) ---")
    print(rules.nlargest(10, 'lift')[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

else:
    print(f"Preprocessed data file not found: {preprocessed_file}")
    print("Please run Preprocessing.py first to generate the preprocessed data.")