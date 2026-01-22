import pandas as pd
import numpy as np
import os
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.model_selection import train_test_split
from Preprocessing import SymptomStandardizer

# Initialize standardizer
standardizer = SymptomStandardizer("symptom_mapping.csv")

# --- 2. Symptom Recommender ---
class SymptomRecommender:
    def __init__(self, rules_df, train_df, standardizer=None):
        self.rules = rules_df
        self.train_df = train_df
        self.standardizer = standardizer
        self.gender_map = {'male': 'M', 'female': 'F', 'm': 'M', 'f': 'F'}

    def get_age_group(self, age):
        if age <= 45: return 'Age_Adult'
        if age <= 60: return 'Age_Middle'
        return 'Age_Senior'

    def recommend_next_symptoms(self, gender, age, current_yes_symptoms, top_n=6):
        g_code = self.gender_map.get(str(gender).lower(), 'M')
        age_grp = self.get_age_group(age)
        
        if self.standardizer:
            std_yes_names = {self.standardizer.get_standard(s) for s in current_yes_symptoms}
        else:
            std_yes_names = {str(s).lower().strip() for s in current_yes_symptoms}
        
        demographic_features = {f"G_{g_code}", age_grp}
        symptom_features = {f"Yes_{s}" for s in std_yes_names}
        full_input_features = demographic_features.union(symptom_features)
        
        recommendations = {}

        # --- Diversified Priority Strategy: Mix results from each method ---
        if not self.rules.empty:
            recommendations_by_method = {}
            
            # 1. Strict Rules (Demographics + Symptoms)
            strict_mask = self.rules['antecedents'].apply(lambda x: 
                any(str(i).startswith('Yes_') for i in x) and 
                any(i.startswith('G_') or i.startswith('Age_') for i in x)
            )
            strict_recs = {}
            for _, row in self.rules[strict_mask].iterrows():
                for item in row['consequents']:
                    if str(item).startswith('Yes_'):
                        s_name = item.replace('Yes_', '')
                        if s_name not in std_yes_names:
                            matches = row['antecedents'].intersection(full_input_features)
                            symptom_matches = [m for m in matches if m.startswith('Yes_')]
                            if len(symptom_matches) > 0:
                                match_count = len(matches)
                                score = row['confidence'] * row['lift'] * (match_count ** 2) * 1.0
                                if s_name not in strict_recs or score > strict_recs[s_name]['score']:
                                    strict_recs[s_name] = {"score": score, "conf": row['confidence'], "lift": row['lift'], "method": "Strict Rule"}
            recommendations_by_method["Strict Rule"] = sorted(strict_recs.items(), key=lambda x: x[1]['score'], reverse=True)
            
            # 2. Age + Symptoms Rules
            age_symptom_mask = self.rules['antecedents'].apply(lambda x: 
                any(str(i).startswith('Yes_') for i in x) and 
                any(i.startswith('Age_') for i in x) and
                not any(i.startswith('G_') for i in x)
            )
            age_recs = {}
            for _, row in self.rules[age_symptom_mask].iterrows():
                for item in row['consequents']:
                    if str(item).startswith('Yes_'):
                        s_name = item.replace('Yes_', '')
                        if s_name not in std_yes_names:
                            matches = row['antecedents'].intersection(full_input_features)
                            symptom_matches = [m for m in matches if m.startswith('Yes_')]
                            if len(symptom_matches) > 0:
                                match_count = len(matches)
                                score = row['confidence'] * row['lift'] * (match_count ** 2) * 1.3
                                if s_name not in age_recs or score > age_recs[s_name]['score']:
                                    age_recs[s_name] = {"score": score, "conf": row['confidence'], "lift": row['lift'], "method": "Age+Symptom Fallback"}
            recommendations_by_method["Age+Symptom Fallback"] = sorted(age_recs.items(), key=lambda x: x[1]['score'], reverse=True)
            
            # 3. Gender + Symptoms Rules
            gender_symptom_mask = self.rules['antecedents'].apply(lambda x: 
                any(str(i).startswith('Yes_') for i in x) and 
                any(i.startswith('G_') for i in x) and
                not any(i.startswith('Age_') for i in x)
            )
            gender_recs = {}
            for _, row in self.rules[gender_symptom_mask].iterrows():
                for item in row['consequents']:
                    if str(item).startswith('Yes_'):
                        s_name = item.replace('Yes_', '')
                        if s_name not in std_yes_names:
                            matches = row['antecedents'].intersection(full_input_features)
                            symptom_matches = [m for m in matches if m.startswith('Yes_')]
                            if len(symptom_matches) > 0:
                                match_count = len(matches)
                                score = row['confidence'] * row['lift'] * (match_count ** 2) * 1.1
                                if s_name not in gender_recs or score > gender_recs[s_name]['score']:
                                    gender_recs[s_name] = {"score": score, "conf": row['confidence'], "lift": row['lift'], "method": "Gender+Symptom Fallback"}
            recommendations_by_method["Gender+Symptom Fallback"] = sorted(gender_recs.items(), key=lambda x: x[1]['score'], reverse=True)
            
            # 4. Symptom Rules
            symptom_mask = self.rules['antecedents'].apply(lambda x: 
                all(str(i).startswith('Yes_') for i in x)
            )
            symptom_recs = {}
            for _, row in self.rules[symptom_mask].iterrows():
                for item in row['consequents']:
                    if str(item).startswith('Yes_'):
                        s_name = item.replace('Yes_', '')
                        if s_name not in std_yes_names:
                            matches = row['antecedents'].intersection(full_input_features)
                            # Must match at least one symptom (not just demographics)
                            symptom_matches = [m for m in matches if m.startswith('Yes_')]
                            if len(symptom_matches) > 0:
                                match_count = len(matches)
                                score = row['confidence'] * row['lift'] * (match_count ** 2) * 0.9
                                if s_name not in symptom_recs or score > symptom_recs[s_name]['score']:
                                    symptom_recs[s_name] = {"score": score, "conf": row['confidence'], "lift": row['lift'], "method": "Symptom Rule"}
            recommendations_by_method["Symptom Rule"] = sorted(symptom_recs.items(), key=lambda x: x[1]['score'], reverse=True)
            
            # Combine all recommendations - just rank by score, no soft caps
            method_counts = {"Strict Rule": 0, "Age+Symptom Fallback": 0, "Gender+Symptom Fallback": 0, "Symptom Rule": 0}
            final_results = []
            
            # Get all candidates across all methods
            all_candidates = []
            for method_name, recs in recommendations_by_method.items():
                for sym, metrics in recs:
                    all_candidates.append((sym, metrics, method_name))
            
            # Sort by score descending and take top N
            all_candidates.sort(key=lambda x: x[1]['score'], reverse=True)
            
            # Take top results without method-based constraints
            used_symbols = set()
            for sym, metrics, method_name in all_candidates:
                if len(final_results) >= top_n:
                    break
                if sym in used_symbols:
                    continue
                
                final_results.append((sym, metrics))
                method_counts[method_name] += 1
                used_symbols.add(sym)
            
            return final_results[:top_n]
        
        # If no rules, return empty list
        return []

# --- 3. Utilities ---
def prepare_basket(row):
    """Create a transaction basket for Apriori mining."""
    items = [f"G_{row['gender_code']}", str(row['age_group'])]
    items.extend([f"Yes_{s}" for s in row['yes_list']])
    return items

# --- 4. Main Flow ---
preprocessed_file = "preprocessed_data.csv"
if os.path.exists(preprocessed_file):
    print(f"Loading preprocessed data from {preprocessed_file}...")
    df = pd.read_csv(preprocessed_file)
    
    # Parse list columns from string representation
    import ast
    for col in ['yes_list', 'no_list', 'idk_list', 'diseases', 'procedures', 'basket']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if pd.notna(x) and isinstance(x, str) else [])
    
    print(f"Data loaded: {len(df)} records")
    
    # Split into train/test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    print("Mining rules from Training Set...")
    te = TransactionEncoder()
    te_ary = te.fit(train_df['basket']).transform(train_df['basket'])
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    print(df_encoded.head())
    
    frequent_itemsets = apriori(df_encoded, min_support=0.003, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10).to_string(index=True))
    print(f"Rules mined: {len(rules)}")
    
    recommender = SymptomRecommender(rules, train_df, standardizer)

    # --- 5. Evaluation Loop (Tracking by Method) ---
    hits = 0
    valid_tests = 0
    
    # Track metrics by method
    hit_by_method = {
        "Strict Rule": {"hits": 0, "confs": [], "lifts": []},
        "Symptom Rule": {"hits": 0, "confs": [], "lifts": []},
        "Age+Symptom Fallback": {"hits": 0, "confs": [], "lifts": []},
        "Gender+Symptom Fallback": {"hits": 0, "confs": [], "lifts": []}
    }

    for _, row in test_df.iterrows():
        if len(row['yes_list']) >= 2:
            input_s = row['yes_list'][:-1]
            ground_truth = row['yes_list'][-1]
            
            results = recommender.recommend_next_symptoms(row['gender_code'], row['age'], input_s)
            
            for sym, metrics in results:
                if sym == ground_truth:
                    method = metrics['method']
                    hit_by_method[method]['hits'] += 1
                    hit_by_method[method]['confs'].append(metrics['conf'])
                    hit_by_method[method]['lifts'].append(metrics['lift'])
                    hits += 1
                    break
            valid_tests += 1

    print("\n--- Evaluation on Test Set ---")
    print(f"Total Hit Rate @ 6: {(hits/valid_tests)*100:.2f}% ({hits}/{valid_tests})")
    
    print("\n--- Detailed Results by Method ---")
    for method, data in hit_by_method.items():
        if data['hits'] > 0:
            avg_conf = np.mean(data['confs']) if data['confs'] else 0
            avg_lift = np.mean(data['lifts']) if data['lifts'] else 0
            print(f"\n{method}:")
            print(f"  Hits: {data['hits']} ({(data['hits']/hits)*100:.1f}% of total hits)")
            print(f"  Avg Confidence: {avg_conf:.4f}")
            print(f"  Avg Lift: {avg_lift:.2f}")
        else:
            print(f"\n{method}: 0 hits")

    # --- 6. Manual Test ---
    print("\n--- Manual Test (with Metrics & Method) ---")
    test_gen, test_age, test_syms = 'male', 60, ['ไอ']
    print(f"Inputs: {test_gen}, {test_age} yrs, {test_syms}")
    
    # Debug: Detailed analysis for this specific query
    g_code = 'M' if test_gen.lower() in ['male', 'm'] else 'F'
    age_grp = 'Age_Adult' if test_age <= 45 else ('Age_Middle' if test_age <= 60 else 'Age_Senior')
    std_input_names = {standardizer.get_standard(s) for s in test_syms}
    demographic_features = {f"G_{g_code}", age_grp}
    symptom_features = {f"Yes_{s}" for s in std_input_names}
    full_input = demographic_features.union(symptom_features)
    
    print(f"\nQuery features: {full_input}")
    print(f"\nDebug - Total rules available: {len(recommender.rules)}")
    
    if not recommender.rules.empty:
        # Check each category and matches for this query
        strict_mask = recommender.rules['antecedents'].apply(lambda x: 
            any(str(i).startswith('Yes_') for i in x) and 
            any(i.startswith('G_') or i.startswith('Age_') for i in x)
        )
        age_symptom_mask = recommender.rules['antecedents'].apply(lambda x: 
            any(str(i).startswith('Yes_') for i in x) and 
            any(i.startswith('Age_') for i in x) and
            not any(i.startswith('G_') for i in x)
        )
        gender_symptom_mask = recommender.rules['antecedents'].apply(lambda x: 
            any(str(i).startswith('Yes_') for i in x) and 
            any(i.startswith('G_') for i in x) and
            not any(i.startswith('Age_') for i in x)
        )
        symptom_mask = recommender.rules['antecedents'].apply(lambda x: 
            all(str(i).startswith('Yes_') for i in x)
        )
        print(f"  Strict Rules (Demographics+Symptoms): {strict_mask.sum()}")
        print(f"  Age+Symptom Rules: {age_symptom_mask.sum()}")
        print(f"  Gender+Symptom Rules: {gender_symptom_mask.sum()}")
        print(f"  Symptom Rules: {symptom_mask.sum()}")
        
        # Check matches for this specific query
        strict_rules = recommender.rules[strict_mask]
        matching_strict = strict_rules[strict_rules['antecedents'].apply(lambda x: len(x.intersection(full_input)) > 0)]
        
        age_rules = recommender.rules[age_symptom_mask]
        matching_age = age_rules[age_rules['antecedents'].apply(lambda x: len(x.intersection(full_input)) > 0)]
        
        gender_rules = recommender.rules[gender_symptom_mask]
        matching_gender = gender_rules[gender_rules['antecedents'].apply(lambda x: len(x.intersection(full_input)) > 0)]
        
        symptom_rules = recommender.rules[symptom_mask]
        matching_symptom = symptom_rules[symptom_rules['antecedents'].apply(lambda x: len(x.intersection(full_input)) > 0)]
        
        print(f"\n  Rules matching this specific query:")
        print(f"    Strict Rules matching: {len(matching_strict)}")
        print(f"    Age+Symptom Rules matching: {len(matching_age)}")
        print(f"    Gender+Symptom Rules matching: {len(matching_gender)}")
        print(f"    Symptom Rules matching: {len(matching_symptom)}")
    
    results = recommender.recommend_next_symptoms(test_gen, test_age, test_syms)
    print(f"\n{'No.':<4} {'Symptom':<20} {'Conf':<8} {'Lift':<8} {'Score':<8} {'Method'}")
    print("-" * 75)
    if len(results) == 0:
        print("NO RECOMMENDATIONS FOUND")
    else:
        for i, (sym, metrics) in enumerate(results, 1):
            print(f"{i:<4} {sym:<20} {metrics['conf']:<8.3f} {metrics['lift']:<8.2f} {metrics['score']:<8.2f} {metrics['method']}")

else:
    print(f"Preprocessed data file not found: {preprocessed_file}")
    print("Please run Preprocessing.py first to generate the preprocessed data.")