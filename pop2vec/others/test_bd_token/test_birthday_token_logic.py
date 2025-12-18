#!/usr/bin/env python3
"""
Standalone test script to verify birthday token insertion logic.
Tests the core logic without requiring SLURM or full data processing.
"""

import torch
import pandas as pd
from datetime import datetime
from pathlib import Path


class SimpleBirthdayTokenTester:
    """Simplified version of BirthdayTokenInserter for testing"""
    
    def __init__(self):
        # Create a minimal vocabulary for testing
        self.token_to_id = {
            '[PAD]': 0,
            '[CLS]': 1,
            '[SEP]': 2,
            'municipality_1': 10,
            'gender_M': 11,
            'month_6': 12,  # June
            'year_1985': 13,
            'event_education': 20,
            'event_income': 21,
            'DEATH': 99,
        }
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        
        # Birthday tokens will be added dynamically
        self.birthday_token_ids = {}
        self.next_token_id = 100
        
        self.sep_id = 2
        self.cls_id = 1
        self.pad_id = 0
        
        print("✓ Initialized test vocabulary")
    
    def _add_birthday_token(self, age: int) -> int:
        """Add a birthday token to vocabulary"""
        token_name = f"BIRTHDAY_YEAR_{age}"
        
        if token_name in self.token_to_id:
            return self.token_to_id[token_name]
        
        new_id = self.next_token_id
        self.next_token_id += 1
        
        self.token_to_id[token_name] = new_id
        self.id_to_token[new_id] = token_name
        self.birthday_token_ids[age] = new_id
        
        print(f"  + Added {token_name} -> ID {new_id}")
        return new_id
    
    def _extract_birth_info(self, background_tokens: torch.Tensor) -> tuple:
        """Extract birth year and month from background tokens"""
        birth_year = None
        birth_month = None
        
        for token_id in background_tokens:
            token_id = int(token_id)
            if token_id in self.id_to_token:
                token_name = self.id_to_token[token_id]
                if isinstance(token_name, str):
                    if token_name.startswith("month_"):
                        try:
                            birth_month = int(token_name.split("_")[1])
                        except (ValueError, IndexError):
                            pass
                    elif token_name.startswith("year_"):
                        try:
                            birth_year = int(token_name.split("_")[1])
                        except (ValueError, IndexError):
                            pass
        
        return (birth_year or 1970, birth_month or 1)
    
    def _calculate_genesis_days(self, birth_year: int, birth_month: int, genesis_year: int = 1970) -> int:
        """Calculate days from genesis date to birth date"""
        genesis_date = datetime(genesis_year, 1, 1)
        birth_date = datetime(birth_year, birth_month, 1)
        days_diff = (birth_date - genesis_date).days
        return days_diff
    
    def _calculate_birthday_date(self, genesis_days: int, age: int) -> int:
        """Calculate the absolute date for a birthday at given age"""
        days_since_birth = int(age * 365.25)
        return genesis_days + days_since_birth
    
    def insert_birthdays(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Insert birthday tokens based on age gaps"""
        tokens = input_ids[0]
        abspos = input_ids[1]
        ages = input_ids[2]
        segments = input_ids[3]
        
        # Find where background ends (first [SEP])
        sep_positions = (tokens == self.sep_id).nonzero(as_tuple=False)
        if len(sep_positions) == 0:
            return input_ids
        
        bg_end = int(sep_positions[0].item())
        
        # Extract birth info
        birth_year, birth_month = self._extract_birth_info(tokens[:bg_end+1])
        genesis_date_days = self._calculate_genesis_days(birth_year, birth_month)
        
        print(f"\n  Birth info: year={birth_year}, month={birth_month}")
        print(f"  Genesis days: {genesis_date_days}")
        
        # Check for death tokens
        death_token_id = self.token_to_id.get('DEATH', None)
        
        new_events = []
        
        # Add background (unchanged)
        for i in range(bg_end + 1):
            new_events.append({
                'token': int(tokens[i]),
                'abspos': int(abspos[i]),
                'age': int(ages[i]),
                'segment': int(segments[i])
            })
        
        last_age = 0
        birthday_insertions = []
        
        # Process tokens after background
        for i in range(bg_end + 1, len(tokens)):
            if tokens[i] == self.pad_id:
                break
            
            # Stop if we hit a death token
            if death_token_id is not None and tokens[i] == death_token_id:
                new_events.append({
                    'token': int(tokens[i]),
                    'abspos': int(abspos[i]),
                    'age': int(ages[i]),
                    'segment': int(segments[i])
                })
                break
            
            current_age = int(ages[i])
            
            # Skip age 0 tokens
            if current_age == 0:
                new_events.append({
                    'token': int(tokens[i]),
                    'abspos': int(abspos[i]),
                    'age': int(ages[i]),
                    'segment': int(segments[i])
                })
                continue
            
            # If we have an age gap, insert birthday tokens
            if current_age > last_age + 1:
                for missing_age in range(last_age + 1, current_age):
                    # Create birthday token if it doesn't exist
                    if missing_age not in self.birthday_token_ids:
                        self._add_birthday_token(missing_age)
                    
                    # Calculate correct birthday date
                    birthday_date = self._calculate_birthday_date(genesis_date_days, missing_age)
                    
                    # Add birthday token
                    new_events.append({
                        'token': self.birthday_token_ids[missing_age],
                        'abspos': birthday_date,
                        'age': missing_age,
                        'segment': 1
                    })
                    
                    # Add [SEP] after birthday
                    new_events.append({
                        'token': self.sep_id,
                        'abspos': birthday_date,
                        'age': missing_age,
                        'segment': 1
                    })
                    
                    birthday_insertions.append((missing_age, birthday_date))
            
            # Add the current event
            new_events.append({
                'token': int(tokens[i]),
                'abspos': int(abspos[i]),
                'age': int(ages[i]),
                'segment': int(segments[i])
            })
            
            # Update last age seen
            if current_age > 0:
                last_age = current_age
        
        # Print birthday insertions
        if birthday_insertions:
            print(f"\n  Birthday tokens inserted:")
            for age, date in birthday_insertions:
                print(f"    - Age {age}: date={date} (token={self.id_to_token[self.birthday_token_ids[age]]})")
        else:
            print(f"\n  No birthday tokens inserted (no age gaps)")
        
        # Convert back to tensor format
        new_len = len(new_events)
        new_input_ids = torch.zeros(4, new_len, dtype=input_ids.dtype)
        
        for i, event in enumerate(new_events):
            new_input_ids[0, i] = event['token']
            new_input_ids[1, i] = event['abspos']
            new_input_ids[2, i] = event['age']
            new_input_ids[3, i] = event['segment']
        
        return new_input_ids


def create_test_sequence(scenario: str) -> torch.Tensor:
    """Create synthetic test sequences for different scenarios"""
    
    if scenario == "age_gap":
        # Scenario: Person born June 1985, has events at age 18, 21, 25
        # Should insert birthdays for ages 1-17, 19-20, 22-24
        input_ids = torch.tensor([
            # [token_id, abspos, age, segment]
            # Background: [CLS], municipality, gender, month, year, [SEP]
            [1, 10, 11, 12, 13, 2,
             # Age 18 event
             20, 
             # Age 21 event
             21,
             # Age 25 event
             20,
             # Padding
             0, 0, 0],
            
            # Absolute positions (days from genesis)
            [0, 0, 0, 0, 0, 0,
             6574,  # Age 18
             7670,  # Age 21
             9131,  # Age 25
             0, 0, 0],
            
            # Ages
            [0, 0, 0, 0, 0, 0,
             18, 21, 25,
             0, 0, 0],
            
            # Segments
            [0, 0, 0, 0, 0, 0,
             1, 1, 1,
             0, 0, 0]
        ], dtype=torch.long)
        
        return input_ids
    
    elif scenario == "no_gap":
        # Scenario: Events at consecutive ages (no gaps)
        input_ids = torch.tensor([
            [1, 10, 11, 12, 13, 2,
             20, 21, 20,
             0, 0, 0],
            [0, 0, 0, 0, 0, 0,
             7305, 7670, 8035,
             0, 0, 0],
            [0, 0, 0, 0, 0, 0,
             20, 21, 22,
             0, 0, 0],
            [0, 0, 0, 0, 0, 0,
             1, 1, 1,
             0, 0, 0]
        ], dtype=torch.long)
        
        return input_ids
    
    elif scenario == "with_death":
        # Scenario: Person dies at age 30, should only insert birthdays up to death
        input_ids = torch.tensor([
            [1, 10, 11, 12, 13, 2,
             20, 99, 21,  # 99 is DEATH token
             0, 0, 0],
            [0, 0, 0, 0, 0, 0,
             7305, 10957, 10957,
             0, 0, 0],
            [0, 0, 0, 0, 0, 0,
             20, 30, 30,
             0, 0, 0],
            [0, 0, 0, 0, 0, 0,
             1, 1, 1,
             0, 0, 0]
        ], dtype=torch.long)
        
        return input_ids
    
    else:
        raise ValueError(f"Unknown scenario: {scenario}")


def print_sequence(input_ids: torch.Tensor, tester: SimpleBirthdayTokenTester, title: str):
    """Pretty print a sequence"""
    print(f"\n{title}")
    print("=" * 80)
    
    tokens = input_ids[0]
    abspos = input_ids[1]
    ages = input_ids[2]
    segments = input_ids[3]
    
    print(f"{'Token':<25} {'ID':>5} {'AbsPos':>10} {'Age':>5} {'Segment':>8}")
    print("-" * 80)
    
    for i in range(len(tokens)):
        token_id = int(tokens[i])
        if token_id == 0:  # Skip padding
            continue
        
        token_name = tester.id_to_token.get(token_id, f"<unknown:{token_id}>")
        abs_pos = int(abspos[i])
        age = int(ages[i])
        segment = int(segments[i])
        
        # Highlight birthday tokens
        if token_name.startswith("BIRTHDAY_YEAR_"):
            token_name = f">>> {token_name} <<<"
        
        print(f"{token_name:<25} {token_id:>5} {abs_pos:>10} {age:>5} {segment:>8}")


def run_test(scenario: str):
    """Run a single test scenario"""
    print(f"\n{'='*80}")
    print(f"TEST SCENARIO: {scenario.upper().replace('_', ' ')}")
    print(f"{'='*80}")
    
    tester = SimpleBirthdayTokenTester()
    
    # Create test sequence
    print(f"\nCreating test sequence for scenario '{scenario}'...")
    input_ids = create_test_sequence(scenario)
    
    # Print original sequence
    print_sequence(input_ids, tester, "ORIGINAL SEQUENCE:")
    
    # Insert birthdays
    print(f"\nProcessing birthday insertions...")
    new_input_ids = tester.insert_birthdays(input_ids)
    
    # Print modified sequence
    print_sequence(new_input_ids, tester, "MODIFIED SEQUENCE (with birthday tokens):")
    
    # Summary statistics
    original_len = (input_ids[0] != 0).sum().item()
    new_len = (new_input_ids[0] != 0).sum().item()
    birthday_count = sum(1 for token_id in new_input_ids[0] 
                        if tester.id_to_token.get(int(token_id), '').startswith('BIRTHDAY_YEAR_'))
    
    print(f"\n{'='*80}")
    print(f"SUMMARY:")
    print(f"  - Original sequence length: {original_len}")
    print(f"  - New sequence length: {new_len}")
    print(f"  - Birthday tokens inserted: {birthday_count}")
    print(f"  - Vocabulary size: {len(tester.token_to_id)}")
    print(f"{'='*80}")


def main():
    """Run all test scenarios"""
    print("\n" + "="*80)
    print("BIRTHDAY TOKEN INSERTION LOGIC TEST")
    print("="*80)
    print("\nThis script tests the birthday token insertion logic using synthetic data.")
    print("It verifies that:")
    print("  1. Birthday tokens are created for ages 1 to N")
    print("  2. Birthday dates are calculated correctly from birth info")
    print("  3. Birthdays are inserted for age gaps in sequences")
    print("  4. Death tokens stop further birthday insertions")
    
    scenarios = [
        "age_gap",      # Most common case: gaps between ages
        "no_gap",       # No gaps: consecutive ages
        "with_death",   # Person dies: should stop inserting after death
    ]
    
    for scenario in scenarios:
        try:
            run_test(scenario)
        except Exception as e:
            print(f"\n❌ TEST FAILED: {scenario}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETE")
    print("="*80)
    print("\nIf you see birthday tokens inserted for age gaps, the logic is working correctly!")
    print("Check that:")
    print("  - Birthday tokens are created dynamically (BIRTHDAY_YEAR_1, BIRTHDAY_YEAR_2, ...)")
    print("  - Dates are calculated correctly based on birth year/month")
    print("  - Birthdays are inserted for ALL missing ages between events")
    print("  - Death tokens stop further birthday insertions")
    print("\n")


if __name__ == "__main__":
    main()
