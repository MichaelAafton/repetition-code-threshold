import unittest
import stim
import pymatching
import numpy as np

from rep_code import (
    build_memory_circuit,
    add_parity_round_general,
    add_round_detectors,
    add_final_detectors
)
from decoder import build_decoder, decode_circuit, count_logical_errors, logical_error_rate

class TestDecoder(unittest.TestCase):
    def test_noiseless_memory(self):
        circuit = build_memory_circuit(d=3, rounds=9, p_data=0.0, p_measurement=0.0)
        matching = build_decoder(circuit)
        detectors, observables, predictions = decode_circuit(circuit, matching, 1000)
        count = count_logical_errors(observables, predictions)
        p_logical = logical_error_rate(count, 1000)
        self.assertEqual(p_logical, 0)

    def test_single_injected_error(self):
        d = 3
        rounds = 9

        # 1. Build the low-noise circuit for the matcher
        matcher_circuit = build_memory_circuit(
            d=3,
            rounds=9,
            p_data=0.01,
            p_measurement=0.01
        )

        matching = build_decoder(matcher_circuit)

        # 2. Construct a separate circuit for the injected error
        circuit = stim.Circuit()

        data_qubits = [2 * i for i in range(d)]
        ancilla_qubits = [2 * j + 1 for j in range(d - 1)]

        circuit.append("R", range(2 * d - 1))

        # 3. Perform measurement rounds and inject the error
        for round_index in range(rounds):
            add_parity_round_general(
                circuit, data_qubits, ancilla_qubits, 0.0
            )

            add_round_detectors(
                circuit, len(ancilla_qubits), round_index == 0
            )

            if round_index == 0:
                circuit.append("X_ERROR", [data_qubits[1]], 1.0)

        # 4. Final measurements and logical observable
        circuit.append("M", data_qubits, 0.0)
        add_final_detectors(circuit, d)
        circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 0)

        # 5. Decode and evaluate
        detectors, observables, predictions = decode_circuit(
            circuit, matching, 1000
        )

        count = count_logical_errors(observables, predictions)

        self.assertEqual(count, 0)

    def test_benchmark_against_stim(self):
        circuit = build_memory_circuit(d=3, rounds=9, p_data=0.05 * 2 / 3, p_measurement=0.05)
        our_matching = build_decoder(circuit)
        ref_circuit = stim.Circuit.generated(
            "repetition_code:memory",
            distance=3,
            rounds=9,
            before_round_data_depolarization=0.05,
            before_measure_flip_probability=0.05
        )
        ref_matching = build_decoder(ref_circuit)
        our_detectors, our_observables, our_predictions = decode_circuit(circuit, our_matching, 200000)
        ref_detectors, ref_observables, ref_predictions = decode_circuit(ref_circuit, ref_matching, 200000)
        our_errors = count_logical_errors(our_observables, our_predictions)
        ref_errors = count_logical_errors(ref_observables, ref_predictions)

        our_rate = logical_error_rate(our_errors, 200000)
        ref_rate = logical_error_rate(ref_errors, 200000)

        print("Our logical error rate:", our_rate)
        print("Stim logical error rate:", ref_rate)
        print("Absolute difference:", abs(our_rate - ref_rate))
        self.assertLess(abs(our_rate - ref_rate), 0.003)
