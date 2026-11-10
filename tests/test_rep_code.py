import unittest
import numpy as np

from rep_code import build_distance_3_memory_circuit


class TestRepetitionCode(unittest.TestCase):

    def test_noiseless_memory(self):
        # build a noiseless circuit
        circuit = build_distance_3_memory_circuit(
            rounds=3,
            p_data=0.0,
            p_measurement=0.0
        )

        # Sample its detectors and observables
        detectors, observables = circuit.compile_detector_sampler().sample(
            shots=5,
            separate_observables=True
        )

        self.assertTrue(np.all(detectors == 0))
        self.assertTrue(np.all(observables == 0))

    def test_measurement_errors(self):
        # Build a circuit with p_measurement = 1.0
        circuit = build_distance_3_memory_circuit(
            rounds=3,
            p_data=0.0,
            p_measurement=1.0
        )

        # Sample its detectors and observables
        detectors, observables = circuit.compile_detector_sampler().sample(
            shots=5,
            separate_observables=True
        )

        # Expected detector pattern for each shot:
        expected = np.array([1, 1, 0, 0, 0, 0, 1, 1])

        # Assert that every shot has this detector pattern
        self.assertTrue(np.all(detectors == expected))
        # Assert that all observables are one
        self.assertTrue(np.all(observables == 1))

    def test_undetectable_logical_error(self):
        # Build a circuit with p_data = 1.0
        circuit = build_distance_3_memory_circuit(
            rounds=3,
            p_data=1.0,
            p_measurement=0.0
        )

        # Sample its detectors and observables
        detectors, observables = circuit.compile_detector_sampler().sample(
            shots=5,
            separate_observables=True
        )

        # Assert that all detectors are zero
        self.assertTrue(np.all(detectors == 0))
        # Assert that all observables are one
        self.assertTrue(np.all(observables == 1))


if __name__ == "__main__":
    unittest.main()
