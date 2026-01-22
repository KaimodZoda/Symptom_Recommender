import pandas as pd
import os
import sys

# Add parent directory to path to import Preprocessing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Preprocessing import SymptomStandardizer

class SymptomRecommender:
    """Symptom recommendation engine using association rules"""
    def __init__(self, rules_df, standardizer):
        self.rules = rules_df
        self.std = standardizer
        self.gender_map = {'male': 'M', 'female': 'F', 'm': 'M', 'f': 'F'}

    def get_age_group(self, age):
        if age <= 45: return 'Age_Adult'
        if age <= 60: return 'Age_Middle'
        return 'Age_Senior'

    def recommend_next_symptoms(self, gender, age, current_yes_symptoms, top_n=6):
        g_code = self.gender_map.get(str(gender).lower(), 'M')
        age_grp = self.get_age_group(age)
        
        if self.std:
            std_yes_names = {self.std.get_standard(s) for s in current_yes_symptoms}
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