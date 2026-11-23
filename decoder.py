import pymatching
import stim
import numpy as np
from rep_code import build_memory_circuit

def build_decoder(circuit):
    dem = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)
    return matching
def decode_circuit(circuit, matching, shots):
    sampler = circuit.compile_detector_sampler()
    detectors, observables = sampler.sample(shots=shots, separate_observables=True)
    predictions = matching.decode_batch(detectors)
    return detectors, observables, predictions
def count_logical_errors(observables, predictions):
    count = np.count_nonzero(observables != predictions)
    return count
def logical_error_rate(logical_errors, shots):
    p_logical = logical_errors/shots
    return p_logical

circuit = build_memory_circuit(d=3, rounds=9, p_data=0.01, p_measurement=0.01)
matching = build_decoder(circuit)
detectors, observables, predictions = decode_circuit(circuit, matching, 10)
