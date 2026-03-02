import pandas as pd
import numpy as np
import os

# i wanted to create a duplicate file instead of overwriting our dataset 
INPUT_FILE = "data/panel_data_digital_finance_clean.csv"
OUTPUT_FILE = "data/panel_data_augmented.csv" 

def augment_dataset(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found. Make sure the script is in the same folder as your CSV.")
        return

    # loading
    df = pd.read_csv(file_path)
    print(f"Original dataset loaded: {len(df)} rows.")

    # to find the last firm ID to start new inputs
    max_id = df['firm_id'].max()
    
    new_entries = []
    
    print(f"Simulating 100 new startups (IDs {max_id + 1} to {max_id + 100})...")
    
    # for generating 100 new inputs
    for i in range(1, 101):
        new_id = max_id + i
        # this for loop is used to simulate new startups (3 years of age) that have 0 funding but can generate revenue
        for year in [2021, 2022, 2023]:
            new_entries.append({
                'firm_id': new_id,
                'year': year,
                'income': round(np.random.uniform(7.0, 9.5), 2),  
                'digital_finance': 0.0,                          
                'firm_size': round(np.random.uniform(0.4, 1.2), 2),
                'employee_num': np.random.randint(1, 6),           
                'founder_edu': np.random.randint(1, 5),            
                'financing_difficulty': round(np.random.uniform(0.85, 1.0), 3),
                'region_gdp': round(np.random.uniform(5.0, 15.0), 2),
                'urban': np.random.choice([0, 1]),
                'high_tech': np.random.choice([0, 1])
            })


    new_data_df = pd.DataFrame(new_entries)
    
    # appending with old dataset
    final_df = pd.concat([df, new_data_df], ignore_index=True)
    
    # used to save the new dataset as a csv
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Success! Augmented dataset saved as: {OUTPUT_FILE}")
    print(f"Total rows now: {len(final_df)} (Added {len(new_entries)} observations).")

if __name__ == "__main__":
    augment_dataset(INPUT_FILE)