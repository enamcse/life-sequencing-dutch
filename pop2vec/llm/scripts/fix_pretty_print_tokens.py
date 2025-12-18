"""
Fix for pretty_print_tokens function to output one line per prefix/generated sequence.
This script shows the corrected version of the function.
"""

from typing import List, Optional
import pandas as pd


def ids_to_tokens(id_list: List[int], vocab_df: pd.DataFrame, with_category: bool = False) -> List[str]:
    """Convert token IDs to human-readable strings."""
    id_to_row = vocab_df.set_index("ID")
    names = []
    for tid in id_list:
        if tid in id_to_row.index:
            row = id_to_row.loc[tid]
            token_name = str(row["TOKEN"])
            if with_category:
                category = str(row["CATEGORY"])
                names.append(f"{token_name}|{category}")
            else:
                names.append(token_name)
        else:
            names.append(f"<UNK:{tid}>")
    return names


def pretty_render_tokens_fixed(
    title: str,
    id_list: List[int],
    vocab_df: pd.DataFrame,
    with_category: bool = False
) -> str:
    """
    Return a nicely formatted string for tokens - ONE LINE per sequence.
    Fixed version that doesn't split long sequences across multiple lines.
    """
    names = ids_to_tokens(id_list, vocab_df, with_category=with_category)
    # Single line: title, count, comma-separated tokens
    return f"{title},{len(id_list)},{','.join(names)}"


def pretty_print_tokens_fixed(
    title: str,
    id_list: List[int],
    vocab_df: pd.DataFrame,
    with_category: bool = False,
    out_path: Optional[str] = None
):
    """
    Print tokens by name; also write to file if out_path is given.
    Fixed version that writes ONE LINE per sequence.
    """
    txt = pretty_render_tokens_fixed(title, id_list, vocab_df, with_category=with_category)
    print(txt)
    if out_path:
        # append so you can print multiple sections to one file
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(txt + "\n")


if __name__ == "__main__":
    print("This file contains the fixed version of pretty_print_tokens.")
    print("To apply this fix, replace the functions in utils.py:")
    print("  - pretty_render_tokens (line ~475)")
    print("  - pretty_print_tokens (line ~483)")
    print("\nThe key change: Remove 'max_per_line' parameter and output everything on one line.")
