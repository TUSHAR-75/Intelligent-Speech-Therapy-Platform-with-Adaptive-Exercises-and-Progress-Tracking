from g2p_en import G2p
import json

g2p = G2p()

# Load sentences
with open("tools/g2p/sentences.txt", "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f if line.strip()]

sentence_phonemes = {}
phoneme_set = set()

for sent in sentences:
    phonemes = g2p(sent)

    # Remove spaces
    phonemes = [p for p in phonemes if p != " "]

    sentence_phonemes[sent] = phonemes

    for p in phonemes:
        # Ignore punctuation tokens
        if p.isalpha():
            phoneme_set.add(p)

# Build inventory
inventory = sorted(list(phoneme_set))

output = {
    "sentences": sentence_phonemes,
    "phoneme_inventory": inventory
}

# Save
with open("tools/g2p/phoneme_inventory.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print("✅ Phoneme inventory built!")
print("Total unique phonemes:", len(inventory))
print(inventory)
