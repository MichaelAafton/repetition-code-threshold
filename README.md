# Repetition code threshold

Simulating the bit-flip repetition code with data and measurement noise in
Stim, decoding with PyMatching (minimum-weight perfect matching), and
measuring how the logical error rate changes with code distance.

I'm doing this to learn quantum error correction hands-on. Next step is the
rotated surface code.

## Files
- `rep_code.py`: builds the circuit, samples, decodes, counts logical errors
- `sweep.py`: runs over distances and error rates, writes `results/rep_code.csv`
- `plot.py`: logical vs physical error rate, with binomial error bars

## Run
    pip install -r requirements.txt
    python sweep.py
    python plot.py

## Status
- [ ] repetition code, d = 3, 5, 7, 9
- [ ] rotated surface code threshold