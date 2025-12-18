# Logging Structure for Birthday Token Insertion

## Overview
The birthday token insertion script has been optimized for clean, organized logging that separates high-level status messages from detailed processing information.

## Logging Destinations

### Standard Output (`.out` file)
Minimal, high-level messages only:
- Job start notification
- Configuration file path
- Job completion status
- Output file location

**Purpose**: Quick job status check without parsing large log files.

### Standard Error (`.err` file via logger)
Detailed processing information:
- Configuration details
- Pre-processing steps (age scanning, vocabulary updates)
- Progress bars (batch processing, writing samples)
- Comprehensive statistics summary
- Error messages (if any)

**Purpose**: Detailed monitoring and debugging. All `logger.info()` calls go here.

## Logging Stages

### 1. Job Start (stdout)
```
Starting birthday token insertion job...
Config: /path/to/config.json
```

### 2. Configuration (stderr/logger)
```
================================================================================
STARTING BIRTHDAY TOKEN INSERTION
================================================================================
Input:  /path/to/input.h5
Output: /path/to/output.h5
Vocab:  /path/to/vocab.csv

CONFIGURATION:
  max_seq_len:  512
  batch_size:   1000
  num_workers:  4
  mlm_encoded:  False
  device:       cpu
================================================================================
```

### 3. Pre-processing (stderr/logger)
- Step 1: Scanning data to find unique ages
- Step 2: Pre-populating vocabulary with birthday tokens
- Step 3: Creating batches and processing in parallel

### 4. Processing (stderr/logger)
- Progress bar for batch processing
- Progress bar for writing samples
- No per-worker or per-token verbose messages

### 5. Statistics Summary (stderr/logger)
```
================================================================================
PROCESSING COMPLETE
================================================================================
Input file:  /path/to/input.h5
Output file: /path/to/output.h5

BIRTHDAY TOKEN INSERTION STATISTICS:
  Total samples processed:         10,000
  Sequences modified:              8,234 (82.3%)
  Sequences unchanged:             1,766 (17.7%)

  Total birthday tokens inserted:  45,678
  Average per sequence (all):      4.57
  Average per modified sequence:   5.55

  Average sequence length (after): 287.3 tokens
  Maximum sequence length:         512

VOCABULARY:
  Birthday tokens in vocabulary:   100 (age 1-100)
  Total vocabulary size:           12,345 tokens
================================================================================
```

### 6. Job Complete (stdout)
```
Birthday token insertion job completed successfully!
Output: /path/to/output.h5
```

## Design Rationale

### Why This Structure?

1. **Separation of Concerns**
   - `.out` file: Quick status check for job submission/completion
   - `.err` file: Detailed logs for monitoring and debugging

2. **Reduced Verbosity**
   - Removed per-worker print statements that flooded logs
   - Removed per-token insertion messages
   - Consolidated progress into clean progress bars

3. **Organized Statistics**
   - All key metrics in one comprehensive summary
   - Clear formatting with thousand separators
   - Percentage calculations for quick insights

4. **Practical Benefits**
   - `.out` files remain small and manageable
   - `.err` files contain all necessary debugging info
   - Easy to grep/search for specific information
   - Clear visual separation with dividers

## Comparison: Before vs. After

### Before (Verbose)
- `.out` file: Thousands of `[Worker] Inserted X birthday tokens...` messages
- Mixed high-level and low-level logging
- Difficult to find relevant information
- Large file sizes

### After (Organized)
- `.out` file: ~4 lines total (start, config, complete, output)
- `.err` file: Organized sections with clear headers
- Statistics summary at the end
- Much smaller file sizes, easier to read

## Usage Tips

### Quick Status Check
```bash
# Check if job completed successfully
tail /path/to/job.out

# Should see:
# Birthday token insertion job completed successfully!
# Output: /path/to/output.h5
```

### Detailed Monitoring
```bash
# Watch processing in real-time
tail -f /path/to/job.err

# Check final statistics
tail -30 /path/to/job.err
```

### Debugging
```bash
# Search for errors
grep -i error /path/to/job.err

# Check configuration
grep -A 10 "CONFIGURATION:" /path/to/job.err

# View statistics
grep -A 20 "PROCESSING COMPLETE" /path/to/job.err
```

## Technical Implementation

### Logger Setup
```python
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
```
- All `logger.info()` calls go to stderr
- Includes timestamp, level, and module name

### Progress Bars (tqdm)
```python
for batch_result in tqdm(pool.imap(process_func, batch_indices),
                         total=len(batch_indices),
                         desc="Processing batches"):
    ...
```
- Visual progress feedback in stderr
- No per-item print statements needed

### Minimal Stdout
```python
print(f"Starting birthday token insertion job...")
print(f"Config: {args.config}")
# ... processing ...
print(f"Birthday token insertion job completed successfully!")
print(f"Output: {output_path}")
```
- Only essential status messages
- Easy to parse programmatically

## Future Improvements

Potential enhancements for more advanced logging:

1. **JSON Logging**: Optional structured logging for automated parsing
2. **Log Levels**: Use DEBUG/INFO/WARNING/ERROR appropriately
3. **Metrics File**: Export statistics to separate JSON/CSV file
4. **Time Tracking**: Add elapsed time measurements for each stage
5. **Resource Monitoring**: Log CPU/memory usage during processing

## Related Files

- Main script: `src/new_code/add_birthday_token_to_preprocess_data.py`
- SLURM scripts: `slurm_scripts/snellius/add_birthday_tokens_*.sh`
- Configs: `configs/Snellius/add_birthdays_*.json`
