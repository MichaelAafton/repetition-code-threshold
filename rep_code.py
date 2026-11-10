import stim

def add_parity_round(circuit, measurement_error_probability=0.0):
    circuit.append("CX", [0, 1, 2, 3])
    circuit.append("CX", [2, 1, 4, 3])
    circuit.append("MR", [1, 3], measurement_error_probability)

def build_distance_3_memory_circuit(
    rounds=3,
    p_data=0.0,
    p_measurement=0.0):

    circuit = stim.Circuit()
    circuit.append("R", [0, 1, 2, 3, 4])

    for round_index in range(rounds):
        add_parity_round(circuit, p_measurement)
        if round_index == 0:
            circuit.append("DETECTOR", [stim.target_rec(-2)])  # A
            circuit.append("DETECTOR", [stim.target_rec(-1)])  # B
        else:
            circuit.append("DETECTOR", [stim.target_rec(-2), stim.target_rec(-4)])
            circuit.append("DETECTOR", [stim.target_rec(-1), stim.target_rec(-3)])
    circuit.append("X_ERROR", [0, 2, 4], p_data)
    circuit.append("M", [0, 2, 4], p_measurement) # noisy final readout
    circuit.append("DETECTOR", [stim.target_rec(-3), stim.target_rec(-2), stim.target_rec(-5)]) # final detector check for A
    circuit.append("DETECTOR", [stim.target_rec(-2), stim.target_rec(-1), stim.target_rec(-4)]) # final detector check for B
    circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 0)
    return circuit

if __name__ == "__main__":
    for name, p_data, p_measurement in [
        ("No noise", 0.0, 0.0),
        ("Measurement errors", 0.0, 1.0),
        ("Logical error", 1.0, 0.0)]:
        circuit = build_distance_3_memory_circuit(
            rounds=3,
            p_data=p_data,
            p_measurement=p_measurement

        )
        detectors, observables = circuit.compile_detector_sampler().sample(shots=3, separate_observables=True)
        print(f"\n{name}")
        print("Detectors:", detectors.astype(int))
        print("Observables:", observables.astype(int))



