import stim

def add_parity_round_general(circuit, data_qubits, ancilla_qubits, measurement_error_probability=0.0):
    first_layer = []
    second_layer = []
    for j in range(len(ancilla_qubits)):
        first_layer.append(data_qubits[j])
        first_layer.append(ancilla_qubits[j])
        second_layer.append(data_qubits[j+1])
        second_layer.append(ancilla_qubits[j])
    circuit.append("CX", first_layer)
    circuit.append("CX", second_layer)
    circuit.append("MR", ancilla_qubits, measurement_error_probability)

def add_round_detectors(circuit, num_ancillas, first_round):
    for j in range(num_ancillas):
        if first_round:
            circuit.append("DETECTOR", [stim.target_rec(j - num_ancillas)])
        else:
            circuit.append("DETECTOR", [stim.target_rec(j - num_ancillas), stim.target_rec(j - 2*num_ancillas)])
    return circuit
def add_final_detectors(circuit, d):
    num_ancillas = d - 1
    for j in range(num_ancillas):
        circuit.append("DETECTOR", [stim.target_rec(j - d), stim.target_rec(j+1 - d), stim.target_rec(j - (num_ancillas + d))])

def build_memory_circuit(
    d,
    rounds=3,
    p_data=0.0,
    p_measurement=0.0
):
    if d < 3 or d % 2 == 0:
        raise ValueError("Distance must be an odd integer >= 3")

    if rounds < 1:
        raise ValueError("Rounds must be at least 1")

    if not 0 <= p_data <= 1 or not 0 <= p_measurement <= 1:
        raise ValueError("Noise probabilities must be between 0 and 1")

    data_qubits = [2 * i for i in range(d)]
    ancilla_qubits = [2 * j + 1 for j in range(d - 1)]

    circuit = stim.Circuit()
    circuit.append("R", range(2 * d - 1))

    for round_index in range(rounds):
        circuit.append("X_ERROR", data_qubits, p_data)
        add_parity_round_general(
            circuit,
            data_qubits,
            ancilla_qubits,
            p_measurement
        )

        add_round_detectors(
            circuit,
            len(ancilla_qubits),
            round_index == 0
        )


    circuit.append("M", data_qubits, p_measurement)

    add_final_detectors(circuit, d)

    circuit.append(
        "OBSERVABLE_INCLUDE",
        [stim.target_rec(-1)],
        0
    )

    return circuit

if __name__ == "__main__":
    for name, p_data, p_measurement in [
        ("No noise", 0.0, 0.0),
        ("Measurement errors", 0.0, 1.0),
        ("Logical error", 1.0, 0.0)]:
        circuit = build_memory_circuit(d=3, rounds=3, p_data=p_data, p_measurement=p_measurement)
        detectors, observables = circuit.compile_detector_sampler().sample(shots=3, separate_observables=True)
        print(f"\n{name}")
        print("Detectors:", detectors.astype(int))
        print("Observables:", observables.astype(int))





