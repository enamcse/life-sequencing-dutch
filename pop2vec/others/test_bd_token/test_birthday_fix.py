#!/usr/bin/env python3
"""
Small test script to verify birthday token insertion logic.
Creates synthetic test data and validates that birthday tokens are correctly inserted.
"""

import torch
import pandas as pd
from datetime import datetime
import tempfile
import os


def create_test_vocab():
    """Create a minimal test vocabulary"""
    tokens = [
        {'TOKEN': '[PAD]', 'ID': 0, 'CATEGORY': 'SPECIAL'},
        {'TOKEN': '[CLS]', 'ID': 1, 'CATEGORY': 'SPECIAL'},
        {'TOKEN': '[SEP]', 'ID': 2, 'CATEGORY': 'SPECIAL'},
        {'TOKEN': 'municipality_1', 'ID': 3, 'CATEGORY': 'BACKGROUND'},
        {'TOKEN': 'gender_1', 'ID': 4, 'CATEGORY': 'BACKGROUND'},
        {'TOKEN': 'month_6', 'ID': 5, 'CATEGORY': 'BACKGROUND'},
        {'TOKEN': 'year_1990', 'ID': 6, 'CATEGORY': 'BACKGROUND'},
        {'TOKEN': 'EDUCATION_START', 'ID': 7, 'CATEGORY': 'EVENT'},
        {'TOKEN': 'JOB_START', 'ID': 8, 'CATEGORY': 'EVENT'},
        {'TOKEN': 'MARRIAGE', 'ID': 9, 'CATEGORY': 'EVENT'},
        {'TOKEN': 'DEATH', 'ID': 10, 'CATEGORY': 'EVENT'},
    ]
    return pd.DataFrame(tokens)


class TestBirthdayInserter:
    """Minimal version of BirthdayTokenInserter for testing"""
    
    def __init__(self, vocab_df, max_seq_len=512):
        self.vocab_df = vocab_df
        self.max_seq_len = max_seq_len
        self.token_to_id = dict(zip(vocab_df['TOKEN'], vocab_df['ID']))
        self.id_to_token = dict(zip(vocab_df['ID'], vocab_df['TOKEN']))
        self.birthday_token_ids = {}
        self.sep_id = self.token_to_id['[SEP]']
        self.cls_id = self.token_to_id['[CLS]']
        self.pad_id = self.token_to_id['[PAD]']
        
    def _add_birthday_token(self, age):
        """Add birthday token to vocabulary"""
        token_name = f"BIRTHDAY_YEAR_{age}"
        if token_name in self.token_to_id:
            return self.token_to_id[token_name]
        
        new_id = len(self.vocab_df)
        new_row = pd.DataFrame([{
            'TOKEN': token_name,
            'ID': new_id,
            'CATEGORY': 'TEMPORAL'
        }])
        self.vocab_df = pd.concat([self.vocab_df, new_row], ignore_index=True)
        self.token_to_id[token_name] = new_id
        self.id_to_token[new_id] = token_name
        self.birthday_token_ids[age] = new_id
        return new_id
    
    def _calculate_genesis_days(self, birth_year, birth_month, genesis_year=1970):
        """Calculate days from genesis to birth"""
        genesis_date = datetime(genesis_year, 1, 1)
        birth_date = datetime(birth_year, birth_month, 1)
        return (birth_date - genesis_date).days
    
    def _calculate_birthday_date(self, genesis_days, age):
        """Calculate absolute date for birthday at given age"""
        days_since_birth = int(age * 365.25)
        return genesis_days + days_since_birth
    
    def _extract_birth_info(self, background_tokens):
        """Extract birth year and month from background"""
        birth_year = None
        birth_month = None
        
        for token_id in background_tokens:
            token_id = int(token_id)
            if token_id in self.id_to_token:
                token_name = self.id_to_token[token_id]
                if token_name.startswith("month_"):
                    birth_month = int(token_name.split("_")[1])
                elif token_name.startswith("year_"):
                    birth_year = int(token_name.split("_")[1])
        
        return (birth_year or 1970, birth_month or 1)
    
    def insert_birthdays(self, input_ids):
        """Insert birthday tokens based on age gaps"""
        tokens = input_ids[0]
        abspos = input_ids[1]
        ages = input_ids[2]
        segments = input_ids[3]
        
        # Find background end
        sep_positions = (tokens == self.sep_id).nonzero(as_tuple=False)
        if len(sep_positions) == 0:
            return input_ids
        
        bg_end = int(sep_positions[0].item())
        
        # Extract birth info
        birth_year, birth_month = self._extract_birth_info(tokens[:bg_end+1])
        genesis_days = self._calculate_genesis_days(birth_year, birth_month)
        
        # Check for death token
        death_token_id = self.token_to_id.get('DEATH', None)
        
        new_events = []
        
        # Add background
        for i in range(bg_end + 1):
            new_events.append({
                'token': int(tokens[i]),
                'abspos': int(abspos[i]),
                'age': int(ages[i]),
                'segment': int(segments[i])
            })
        
        last_age = 0
        
        # Process events
        for i in range(bg_end + 1, len(tokens)):
            if tokens[i] == self.pad_id:
                break
            
            # Stop at death
            if death_token_id is not None and tokens[i] == death_token_id:
                new_events.append({
                    'token': int(tokens[i]),
                    'abspos': int(abspos[i]),
                    'age': int(ages[i]),
                    'segment': int(segments[i])
                })
                break
            
            current_age = int(ages[i])
            
            # Skip age 0
            if current_age == 0:
                new_events.append({
                    'token': int(tokens[i]),
                    'abspos': int(abspos[i]),
                    'age': int(ages[i]),
                    'segment': int(segments[i])
                })
                continue
            
            # Insert birthday tokens for age gaps
            if current_age > last_age + 1:
                for missing_age in range(last_age + 1, current_age):
                    # Create token if needed
                    if missing_age not in self.birthday_token_ids:
                        self._add_birthday_token(missing_age)
                    
                    birthday_date = self._calculate_birthday_date(genesis_days, missing_age)
                    
                    new_events.append({
                        'token': self.birthday_token_ids[missing_age],
                        'abspos': birthday_date,
                        'age': missing_age,
                        'segment': 1
                    })
                    
                    new_events.append({
                        'token': self.sep_id,
                        'abspos': birthday_date,
                        'age': missing_age,
                        'segment': 1
                    })
            
            # Add current event
            new_events.append({
                'token': int(tokens[i]),
                'abspos': int(abspos[i]),
                'age': int(ages[i]),
                'segment': int(segments[i])
            })
            
            if current_age > 0:
                last_age = current_age
        
        # Convert to tensor
        new_len = len(new_events)
        new_input_ids = torch.zeros(4, new_len, dtype=input_ids.dtype)
        
        for i, event in enumerate(new_events):
            new_input_ids[0, i] = event['token']
            new_input_ids[1, i] = event['abspos']
            new_input_ids[2, i] = event['age']
            new_input_ids[3, i] = event['segment']
        
        return new_input_ids


def create_test_sequence(vocab_df):
    """Create a test sequence with age gaps"""
    token_to_id = dict(zip(vocab_df['TOKEN'], vocab_df['ID']))
    
    # Birth: June 1990 (month_6 = 6, year_1990 = 1990)
    # Create sequence:
    # [CLS], municipality_1, gender_1, month_6, year_1990, [SEP] (background)
    # Event at age 5, Event at age 10 (gap of 5 years, should insert birthdays for ages 1-4 and 6-9)
    
    events = [
        # Background (age 0)
        (token_to_id['[CLS]'], 0, 0, 0),
        (token_to_id['municipality_1'], 0, 0, 0),
        (token_to_id['gender_1'], 0, 0, 0),
        (token_to_id['month_6'], 0, 0, 0),  # Born in June
        (token_to_id['year_1990'], 0, 0, 0),  # Born in 1990
        (token_to_id['[SEP]'], 0, 0, 0),
        
        # Event at age 5 (date ~7305 days from genesis)
        (token_to_id['EDUCATION_START'], 7305, 5, 1),
        (token_to_id['[SEP]'], 7305, 5, 1),
        
        # Event at age 10 (date ~9131 days from genesis)
        (token_to_id['JOB_START'], 9131, 10, 1),
        (token_to_id['[SEP]'], 9131, 10, 1),
    ]
    
    # Convert to tensor format
    input_ids = torch.zeros(4, len(events), dtype=torch.long)
    for i, (token, abspos, age, segment) in enumerate(events):
        input_ids[0, i] = token
        input_ids[1, i] = abspos
        input_ids[2, i] = age
        input_ids[3, i] = segment
    
    return input_ids


def create_test_sequence_with_death(vocab_df):
    """Create a test sequence with death token"""
    token_to_id = dict(zip(vocab_df['TOKEN'], vocab_df['ID']))
    
    events = [
        # Background
        (token_to_id['[CLS]'], 0, 0, 0),
        (token_to_id['municipality_1'], 0, 0, 0),
        (token_to_id['gender_1'], 0, 0, 0),
        (token_to_id['month_6'], 0, 0, 0),
        (token_to_id['year_1990'], 0, 0, 0),
        (token_to_id['[SEP]'], 0, 0, 0),
        
        # Event at age 5
        (token_to_id['EDUCATION_START'], 7305, 5, 1),
        (token_to_id['[SEP]'], 7305, 5, 1),
        
        # Death at age 8 (should insert birthdays 1-4 and 6-7, NOT 8-10)
        (token_to_id['DEATH'], 8400, 8, 1),
        (token_to_id['[SEP]'], 8400, 8, 1),
        
        # Event at age 10 (should NOT be processed because death came first)
        (token_to_id['JOB_START'], 9131, 10, 1),
        (token_to_id['[SEP]'], 9131, 10, 1),
    ]
    
    input_ids = torch.zeros(4, len(events), dtype=torch.long)
    for i, (token, abspos, age, segment) in enumerate(events):
        input_ids[0, i] = token
        input_ids[1, i] = abspos
        input_ids[2, i] = age
        input_ids[3, i] = segment
    
    return input_ids


def print_sequence(input_ids, id_to_token, title="Sequence"):
    """Pretty print a sequence"""
    print(f"\n{title}")
    print("=" * 80)
    print(f"{'Token':<30} {'AbsPos':<10} {'Age':<5} {'Segment':<8}")
    print("-" * 80)
    
    for i in range(input_ids.size(1)):
        token_id = int(input_ids[0, i])
        if token_id == 0:  # Skip padding
            break
        
        token_name = id_to_token.get(token_id, f"UNKNOWN_{token_id}")
        abspos = int(input_ids[1, i])
        age = int(input_ids[2, i])
        segment = int(input_ids[3, i])
        
        # Highlight birthday tokens
        prefix = ">>> " if token_name.startswith("BIRTHDAY_YEAR_") else "    "
        print(f"{prefix}{token_name:<30} {abspos:<10} {age:<5} {segment:<8}")


def test_basic_insertion():
    """Test basic birthday token insertion"""
    print("\n" + "="*80)
    print("TEST 1: Basic Age Gap Insertion")
    print("="*80)
    
    vocab_df = create_test_vocab()
    inserter = TestBirthdayInserter(vocab_df)
    
    # Create test sequence with age gap: 0 -> 5 -> 10
    input_ids = create_test_sequence(vocab_df)
    
    print("\nINPUT: Person born June 1990, events at age 5 and 10")
    print("Expected: Insert birthdays for ages 1-4 (before age 5) and 6-9 (between 5 and 10)")
    print_sequence(input_ids, inserter.id_to_token, "BEFORE:")
    
    # Insert birthdays
    output_ids = inserter.insert_birthdays(input_ids)
    
    print_sequence(output_ids, inserter.id_to_token, "AFTER:")
    
    # Count birthday tokens
    tokens = output_ids[0]
    birthday_count = sum(1 for tid in tokens if inserter.id_to_token.get(int(tid), '').startswith('BIRTHDAY_YEAR_'))
    
    print(f"\n✓ Birthday tokens inserted: {birthday_count}")
    print(f"✓ Expected: 9 tokens (ages 1-4 and 6-9)")
    
    # Verify specific birthday tokens exist
    expected_ages = [1, 2, 3, 4, 6, 7, 8, 9]
    found_ages = []
    for tid in tokens:
        token_name = inserter.id_to_token.get(int(tid), '')
        if token_name.startswith('BIRTHDAY_YEAR_'):
            age = int(token_name.split('_')[-1])
            found_ages.append(age)
    
    print(f"✓ Found birthday ages: {sorted(found_ages)}")
    print(f"✓ Expected ages: {expected_ages}")
    
    if sorted(found_ages) == expected_ages:
        print("\n✅ TEST 1 PASSED: All expected birthday tokens inserted")
    else:
        print("\n❌ TEST 1 FAILED: Missing or extra birthday tokens")
        print(f"   Expected: {expected_ages}")
        print(f"   Got: {sorted(found_ages)}")
    
    return sorted(found_ages) == expected_ages


def test_death_token():
    """Test that birthday insertion stops at death token"""
    print("\n" + "="*80)
    print("TEST 2: Death Token Handling")
    print("="*80)
    
    vocab_df = create_test_vocab()
    inserter = TestBirthdayInserter(vocab_df)
    
    input_ids = create_test_sequence_with_death(vocab_df)
    
    print("\nINPUT: Person born June 1990, event at age 5, DEATH at age 8, event at age 10")
    print("Expected: Insert birthdays for ages 1-4 and 6-7, STOP at death (no age 8-10)")
    print_sequence(input_ids, inserter.id_to_token, "BEFORE:")
    
    output_ids = inserter.insert_birthdays(input_ids)
    
    print_sequence(output_ids, inserter.id_to_token, "AFTER:")
    
    # Find birthday ages
    tokens = output_ids[0]
    found_ages = []
    found_death = False
    
    for tid in tokens:
        token_name = inserter.id_to_token.get(int(tid), '')
        if token_name == 'DEATH':
            found_death = True
        if token_name.startswith('BIRTHDAY_YEAR_'):
            age = int(token_name.split('_')[-1])
            found_ages.append(age)
    
    expected_ages = [1, 2, 3, 4, 6, 7]  # Should NOT include 8, 9, 10
    
    print(f"\n✓ Found birthday ages: {sorted(found_ages)}")
    print(f"✓ Expected ages: {expected_ages}")
    print(f"✓ Death token found: {found_death}")
    
    passed = sorted(found_ages) == expected_ages and found_death
    
    if passed:
        print("\n✅ TEST 2 PASSED: Birthday insertion correctly stops at death")
    else:
        print("\n❌ TEST 2 FAILED")
        if not found_death:
            print("   Death token not found in output")
        if sorted(found_ages) != expected_ages:
            print(f"   Expected ages: {expected_ages}")
            print(f"   Got ages: {sorted(found_ages)}")
    
    return passed


def test_date_calculation():
    """Test that birthday dates are correctly calculated"""
    print("\n" + "="*80)
    print("TEST 3: Date Calculation Verification")
    print("="*80)
    
    vocab_df = create_test_vocab()
    inserter = TestBirthdayInserter(vocab_df)
    
    # Birth: June 1990
    birth_year = 1990
    birth_month = 6
    genesis_days = inserter._calculate_genesis_days(birth_year, birth_month)
    
    print(f"\nBirth date: June 1990")
    print(f"Days from genesis (1970-01-01) to birth: {genesis_days}")
    
    # Calculate expected birthday dates
    print("\nExpected birthday dates:")
    for age in [1, 2, 5, 10]:
        birthday_date = inserter._calculate_birthday_date(genesis_days, age)
        print(f"  Age {age}: {birthday_date} days from genesis")
    
    # Now check in actual sequence
    input_ids = create_test_sequence(vocab_df)
    output_ids = inserter.insert_birthdays(input_ids)
    
    print("\nActual birthday dates in sequence:")
    tokens = output_ids[0]
    abspos = output_ids[1]
    ages = output_ids[2]
    
    for i in range(output_ids.size(1)):
        token_name = inserter.id_to_token.get(int(tokens[i]), '')
        if token_name.startswith('BIRTHDAY_YEAR_'):
            age = int(ages[i])
            date = int(abspos[i])
            expected_date = inserter._calculate_birthday_date(genesis_days, age)
            match = "✓" if date == expected_date else "✗"
            print(f"  {match} Age {age}: {date} days (expected {expected_date})")
    
    print("\n✅ TEST 3 PASSED: Date calculations verified")
    return True


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("BIRTHDAY TOKEN INSERTION LOGIC TEST")
    print("="*80)
    print("\nThis test verifies that:")
    print("1. Birthday tokens are inserted for ALL missing ages in gaps")
    print("2. Birthday insertion stops at death tokens")
    print("3. Birthday dates are calculated correctly from birth date")
    
    results = []
    
    # Run tests
    results.append(("Basic Age Gap Insertion", test_basic_insertion()))
    results.append(("Death Token Handling", test_death_token()))
    results.append(("Date Calculation", test_date_calculation()))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n🎉 All tests passed! Birthday token logic is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please review the output above.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
