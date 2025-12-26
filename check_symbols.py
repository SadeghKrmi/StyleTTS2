import re

try:
    with open('meldataset.py', 'r', encoding='utf-8') as f:
        content = f.read()

    match = re.search(r'_letters_ipa = "(.*)"', content)
    if match:
        ipa_symbols = match.group(1)
        # Reconstruct the full symbols list logic from meldataset.py
        # symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
        _pad = "$"
        _punctuation = ';:,.!?¡¿—…"«»“” '
        _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        
        symbols = set(_pad) | set(_punctuation) | set(_letters) | set(ipa_symbols)
        
        print(f"Loaded {len(symbols)} symbols from meldataset.py")
        
        used_chars = set()
        with open('Data/train_list.txt', 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.split('|')
                if len(parts) >= 2:
                    text = parts[1]
                    used_chars.update(text)
        
        missing = sorted(list(used_chars - symbols))
        print(f"Missing symbols ({len(missing)}):")
        for char in missing:
            print(f"Character: '{char}' (U+{ord(char):04X})")
            
    else:
        print("Could not find _letters_ipa in meldataset.py")

except Exception as e:
    print(f"Error: {e}")
