# H5 File Analysis Summary

## Life Sequencing Dataset Analysis

### Files Analyzed
- **Main Dataset**: `encoded.h5` (3.14 GB) - 4,441,429 life sequences
- **Test Dataset**: `dryrun_encoded.h5` (7.02 MB) - 10,000 life sequences (subset for testing)

---

## Dataset Structure Overview

### Core Data Format
Each life sequence is represented as a **4-dimensional tensor** with shape `(N, 4, 512)`:
- **N**: Number of life sequences 
- **4**: Four different feature dimensions (explained below)
- **512**: Maximum sequence length (padded)

---

## The Four Dimensions Explained

### **Dimension 0: Original Event Token IDs**
- **Purpose**: Life event identifiers from vocabulary
- **Range**: 0 to 11,652 
- **Vocabulary Size**: ~10,832 unique life events
- **Zero Percentage**: 66.12% (padding)
- **Example**: `[1, 551, 5, 97, 53, 2, 807, 908, ...]`
- **Interpretation**: Each number represents a specific life event (birth, marriage, job change, etc.)

### **Dimension 1: Days Since Genesis Date**
- **Purpose**: Temporal information - days elapsed from a reference date
- **Range**: 0 to 16,438 days (~45 years span)
- **Vocabulary Size**: ~6,330 unique day counts
- **Zero Percentage**: 67.30% (padding)
- **Example**: `[0, 0, 0, 0, 0, 0, 3128, 3128, 3128, 5254, ...]`
- **Interpretation**: Provides precise timing of when each life event occurred

### **Dimension 2: Individual's Age (Years)**
- **Purpose**: Person's age when the event occurred
- **Range**: 0 to 99 years
- **Vocabulary Size**: 85 unique ages
- **Zero Percentage**: 67.30% (padding)
- **Example**: `[0, 0, 0, 0, 0, 0, 49, 49, 49, 43, ...]`
- **Interpretation**: Age context for each life event (e.g., married at 49, had child at 43)

### **Dimension 3: Event Continuation Flag**
- **Purpose**: Indicates if this is the same event as the previous one
- **Range**: 0, 1, or 2 (only 3 values)
- **Zero Percentage**: 67.30% (padding)
- **Values**:
  - `0`: Padding/no event
  - `1`: New/different event from previous
  - `2`: Continuation of previous event
- **Example**: `[0, 0, 0, 0, 0, 0, 2, 2, 2, 2, ...]`

---

## Additional Datasets

### **`original_sequence`** (N × 512)
- **Purpose**: Raw life event tokens (same as Dimension 0)
- **Statistics**: Identical to input_ids dimension 0
- **Usage**: Reference/validation dataset

### **`padding_mask`** (N × 512)
- **Purpose**: Binary mask indicating real data vs padding
- **Values**: 1 = real data, 0 = padding
- **Real Data**: ~34% of positions contain actual life events
- **Usage**: Helps models ignore padded positions during training

### **`sequence_id`** (N,)
- **Purpose**: Unique identifier for each life sequence/person
- **Range**: 220,779 to 999,859,260
- **Usage**: Links sequences to individuals for analysis

---

## Key Insights

### **Multi-Modal Life Representation**
This dataset provides a rich, multi-dimensional view of life sequences:
1. **What happened** (event tokens)
2. **When it happened** (absolute time)
3. **Life stage context** (age)
4. **Event structure** (continuation patterns)

### **Temporal Richness**
- **Dual time representation**: Both absolute time (days since genesis) and relative time (age)
- **Event duration modeling**: Continuation flags capture extended events
- **Life span coverage**: 0-99 years, ~45 years of calendar time

### **Scale and Scope**
- **Large scale**: 4.4M life sequences 
- **Comprehensive**: Average ~34% of 512 positions filled per sequence
- **Standardized**: All sequences normalized to 512 tokens with padding

### **Use Cases**
This data structure is ideal for:
- **Life trajectory prediction**: Predicting future life events
- **Pattern mining**: Finding common life patterns across populations
- **Demographic analysis**: Understanding life events by age/time
- **Event duration modeling**: Analyzing how long life states persist
- **Sequence-to-sequence learning**: Training transformers on life data

---

## Technical Specifications

| Metric | Main Dataset | Test Dataset |
|--------|--------------|--------------|
| **File Size** | 3.14 GB | 7.02 MB |
| **Sequences** | 4,441,429 | 10,000 |
| **Sequence Length** | 512 tokens | 512 tokens |
| **Dimensions** | 4 feature types | 4 feature types |
| **Data Type** | int64 | int64 |
| **Padding Ratio** | ~66% | ~66% |

This represents a sophisticated, temporally-aware life sequence dataset designed for advanced machine learning applications in demographic and social science research.
