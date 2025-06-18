import pandas as pd

vocab_path = "pipe_test_eh/step5/vocab_v0.csv"
vocab = pd.read_csv(vocab_path)

# Reverse map: ID → Token
id_to_token = dict(zip(vocab["ID"], vocab["TOKEN"]))

# Inspect decoded targets
i = 2314
positions = f["target_pos"][i]
tokens = f["target_tokens"][i]

for pos, tok in zip(positions, tokens):
    if pos != -1 and tok != -1:
        token_str = id_to_token.get(tok, "[UNKNOWN]")
        print(f"At pos {pos:3d}: token ID = {tok} → {token_str}")
