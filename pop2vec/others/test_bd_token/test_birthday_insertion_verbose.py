#!/usr/bin/env python3
"""
Test script to verify birthday token insertion logic with verbose logging.
This creates a minimal example showing the insertion logic in action.
"""

try:
    import torch
    import pandas as pd
    from datetime import datetime
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure torch and pandas are installed")
    import sys
    sys.exit(1)

# Simulate a mini vocabulary
vocab_data = {
    'TOKEN': ['[PAD]', '[CLS]', '[SEP]', 'municipality_1', 'gender_M', 
              'month_6', 'year_1985', 'EVENT_A', 'EVENT_B', 'EVENT_C'],
    'ID': list(range(10)),
    'CATEGORY': ['SPECIAL'] * 3 + ['DEMOGRAPHIC'] * 4 + ['EVENT'] * 3
}
vocab_df = pd.DataFrame(vocab_data)

# Add birthday tokens for ages 1-100
birthday_tokens = {}
for age in range(1, 101):
    token_name = f'BIRTHDAY_YEAR_{age}'
    token_id = len(vocab_df)
    vocab_df = pd.concat([vocab_df, pd.DataFrame({
        'TOKEN': [token_name],
        'ID': [token_id],
        'CATEGORY': ['TEMPORAL']
    })], ignore_index=True)
    birthday_tokens[age] = token_id

print(f"Created vocabulary with {len(vocab_df)} tokens")
print(f"Added birthday tokens for ages 1-100: {min(birthday_tokens.values())} to {max(birthday_tokens.values())}")

# Create a test sequence
# Format: [CLS, municipality, gender, month, year, SEP, event@age16, event@age18, event@age20]
input_ids = torch.tensor([
    [1, 3, 4, 5, 6, 2, 7, 8, 9],        # Tokens: CLS, muni, gender, month, year, SEP, EVENT_A, EVENT_B, EVENT_C
    [0, 0, 0, 0, 0, 0, 5500, 6200, 7300],  # Absolute position (days)
    [0, 0, 0, 0, 0, 0, 16, 18, 20],    # Age: background is 0, then 16, 18, 20
    [0, 0, 0, 0, 0, 0, 1, 1, 1]        # Segment: 0 for background, 1 for events
])

print("\n--- Original Sequence ---")
print("Tokens:", input_ids[0].tolist())
print("Abspos:", input_ids[1].tolist())
print("Ages:  ", input_ids[2].tolist())
print("Segments:", input_ids[3].tolist())

# Extract birth info
birth_year = 1985
birth_month = 6
print(f"\nBirth info: {birth_year}-{birth_month:02d}")

# Calculate genesis days
genesis_date = datetime(1970, 1, 1)
birth_date = datetime(birth_year, birth_month, 1)
genesis_days = (birth_date - genesis_date).days
print(f"Genesis days (from 1970-01-01 to birth): {genesis_days}")

# Simulate birthday token insertion
print("\n--- Inserting Birthday Tokens ---")

sep_id = 2
new_events = []

# Add background (first 6 tokens)
for i in range(6):
    new_events.append({
        'token': int(input_ids[0, i]),
        'abspos': int(input_ids[1, i]),
        'age': int(input_ids[2, i]),
        'segment': int(input_ids[3, i])
    })

last_age = 0

# Process events after background
for i in range(6, 9):
    current_age = int(input_ids[2, i])
    
    # Check for age gaps
    if current_age > last_age + 1:
        print(f"\nAge gap detected: {last_age} -> {current_age}")
        print(f"  Inserting birthday tokens for ages: {list(range(last_age + 1, current_age))}")
        
        # Insert birthday tokens for missing ages
        for missing_age in range(last_age + 1, current_age):
            # Calculate birthday date
            days_since_birth = int(missing_age * 365.25)
            birthday_date = genesis_days + days_since_birth
            
            print(f"    Age {missing_age}: BIRTHDAY_YEAR_{missing_age} (ID {birthday_tokens[missing_age]}) at date {birthday_date}")
            
            # Add birthday token
            new_events.append({
                'token': birthday_tokens[missing_age],
                'abspos': birthday_date,
                'age': missing_age,
                'segment': 1
            })
            
            # Add SEP
            new_events.append({
                'token': sep_id,
                'abspos': birthday_date,
                'age': missing_age,
                'segment': 1
            })
    
    # Add current event
    new_events.append({
        'token': int(input_ids[0, i]),
        'abspos': int(input_ids[1, i]),
        'age': int(input_ids[2, i]),
        'segment': int(input_ids[3, i])
    })
    print(f"\n  Added original event at age {current_age}")
    
    last_age = current_age

# Convert to tensor
new_len = len(new_events)
new_input_ids = torch.zeros(4, new_len, dtype=torch.long)

for i, event in enumerate(new_events):
    new_input_ids[0, i] = event['token']
    new_input_ids[1, i] = event['abspos']
    new_input_ids[2, i] = event['age']
    new_input_ids[3, i] = event['segment']

print("\n--- Modified Sequence ---")
print("Tokens:", new_input_ids[0].tolist())
print("Abspos:", new_input_ids[1].tolist())
print("Ages:  ", new_input_ids[2].tolist())
print("Segments:", new_input_ids[3].tolist())

print(f"\n--- Summary ---")
print(f"Original sequence length: {input_ids.shape[1]}")
print(f"Modified sequence length: {new_input_ids.shape[1]}")
print(f"Birthday tokens inserted: {(new_input_ids.shape[1] - input_ids.shape[1]) // 2}")  # Divide by 2 because we add SEP too

# Verify ages are correct
ages_in_sequence = new_input_ids[2][new_input_ids[2] > 0].unique().tolist()
print(f"Ages present in modified sequence: {sorted(ages_in_sequence)}")
print(f"Expected: All ages from 1 to 20 (birth to last event)")
print(f"Missing ages: {set(range(1, 21)) - set(ages_in_sequence)}")
print(f"✓ All ages from 1 to 20 are present!" if set(range(1, 21)) == set(ages_in_sequence) else "✗ Some ages are missing!")
