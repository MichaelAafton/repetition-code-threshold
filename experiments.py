from statistics import NormalDist
import math
from matplotlib.patheffects import Normal
from networkx import config
import csv
import json
import numpy as np

from pathlib import Path

from rep_code import build_memory_circuit
from decoder import build_decoder, decode_circuit, count_logical_errors, logical_error_rate


def run_experiment(d, rounds, p_data, p_measurement, shots):
    circuit = build_memory_circuit(d, rounds, p_data, p_measurement)
    matching = build_decoder(circuit)
    det, obs, pred = decode_circuit(circuit, matching, shots)
    count = count_logical_errors(obs, pred)
    p_logical = logical_error_rate(count, shots)
    lower, upper = calculate_wilson_interval(count, shots, confidence=0.95)
    p_logical_round = calculate_per_round_error_rate(p_logical, rounds)
    wilson_round_lower = calculate_per_round_error_rate(lower, rounds)
    wilson_round_upper = calculate_per_round_error_rate(upper, rounds)
    return {"distance": d,
            "rounds": rounds,
            "p_data": p_data,
            "p_measurement": p_measurement,
            "shots": shots,
            "logical_errors": count,
            "p_logical_total": p_logical,
            "wilson_lower": lower,
            "wilson_upper":  upper,
            "p_logical_round": p_logical_round,
            "wilson_round_lower": wilson_round_lower,
            "wilson_round_upper": wilson_round_upper}

def calculate_wilson_interval(logical_errors, shots, confidence=0.95):
    if shots <= 0:
        raise ValueError("shots must be positive")

    if not 0 <= logical_errors <= shots:
        raise ValueError("logical_errors must be between 0 and shots")

    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    p_hat = logical_errors/shots
    x = 1 - (1 - confidence) / 2
    z = NormalDist().inv_cdf(x)

    denominator = 1.0 + z**2 / shots

    centre = p_hat + z**2 / (2.0 * shots)
    variance_term = (p_hat * (1.0 - p_hat)) / shots + z**2 / (4.0 * shots * shots)
    margin = z * math.sqrt(variance_term)

    lower = (centre - margin) / denominator
    upper = (centre + margin) / denominator
    return (lower, upper)

def calculate_per_round_error_rate(p_total, rounds):
    if not 0 <= p_total <= 0.5:
        raise ValueError("p_total must be between 0 and 0.5")
    if rounds <= 0 and isinstance(rounds, int):
        raise ValueError("There can't be less than 1 round")
    p_round = (1-(1-2*p_total)**(1/rounds))/2
    return p_round

def calculate_relative_uncertainty(logical_errors, shots, confidence=0.95):
    p_hat = logical_errors/shots
    if logical_errors == 0:
        return math.inf
    lower, upper = calculate_wilson_interval(logical_errors, shots, confidence)
    delta_p = (upper - lower)/2
    r = delta_p/p_hat
    return r

def calculate_required_shots(p_estimate, target_relative_error, confidence=0.95):
    if not p_estimate > 0:
        raise ValueError("p_estimate must be greater than 0")
    if  target_relative_error < 0:
        raise ValueError("Target relative error must be greater than 0")
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between 0 and 1")
    x = 1 - (1 - confidence) / 2
    z = NormalDist().inv_cdf(x)
    # Find approximate shot count
    N = ((z**2)*(1-p_estimate))/(p_estimate*target_relative_error**2)
    return math.ceil(N)

def run_adaptive_experiment(
    d, rounds, p_data, p_measurement,
    pilot_shots, target_relative_error,
    confidence=0.95, max_pilot_shots=10000):
    pilot_result = run_experiment(
        d, rounds, p_data, p_measurement, pilot_shots)

    total_pilot_shots = pilot_shots
    logical_errors = pilot_result["logical_errors"]

    while logical_errors == 0 and pilot_shots < max_pilot_shots:
        pilot_shots = min(pilot_shots * 2, max_pilot_shots)

        pilot_result = run_experiment(
            d, rounds, p_data, p_measurement, pilot_shots
        )

        total_pilot_shots += pilot_shots
        logical_errors = pilot_result["logical_errors"]

    est_total_error_rate = pilot_result["p_logical_total"]

    if logical_errors == 0:
        result = pilot_result.copy()
        result.update({
            "status": "pilot_insufficient",
            "pilot_shots": pilot_shots,
            "pilot_total_shots": total_pilot_shots,
            "pilot_error": logical_errors,
            "pilot_p_estimate": est_total_error_rate,
            "production_shots": 0,
            "total_shots": total_pilot_shots,
            "relative_uncertainty": None,
            "target_achieved": False
        })
        return result

    # Calculate required production shots
    required_shots = calculate_required_shots(
        est_total_error_rate, target_relative_error, confidence
    )

    # Independent production experiment
    re1 = run_experiment(
        d, rounds, p_data, p_measurement, required_shots
    )

    production_errors = re1["logical_errors"]

    relative_unc = calculate_relative_uncertainty(
        production_errors, required_shots, confidence
    )

    total_shots = total_pilot_shots + required_shots
    target_achieved = relative_unc <= target_relative_error

    result = re1.copy()
    result.update({
        "status": "complete",
        "pilot_shots": pilot_shots,
        "pilot_total_shots": total_pilot_shots,
        "pilot_error": logical_errors,
        "pilot_p_estimate": est_total_error_rate,
        "production_shots": required_shots,
        "total_shots": total_shots,
        "relative_uncertainty": relative_unc,
        "target_achieved": target_achieved
    })

    return result

def run_parameter_sweep(
    distances, physical_error_rates,
    pilot_shots, target_relative_error,
    confidence=0.95):
    results = []
    for d in distances:
        for p in physical_error_rates:
            result = run_adaptive_experiment(d, d, p, p, pilot_shots, target_relative_error, confidence)
            results.append(result)
    return results

from collections import defaultdict


def calculate_error_suppression(results: list[dict]) -> list[dict]:
    # 1. Group completed results with nonzero per-round logical error rate by p_data
    by_p_data: dict[float, dict[int, float]] = defaultdict(dict)

    for entry in results:
        if entry.get("status") != "complete":
            continue

        p_logical_round = entry.get("p_logical_round")
        if p_logical_round is None or p_logical_round <= 0:
            continue

        p_data = entry["p_data"]
        distance = entry["distance"]
        by_p_data[p_data][distance] = float(p_logical_round)

    suppression_records: list[dict] = []

    # 2–5. Match consecutive distances (d, d + 2) within each physical error rate group
    for p_data in sorted(by_p_data):
        by_distance = by_p_data[p_data]
        for distance in sorted(by_distance):
            next_distance = distance + 2
            if next_distance not in by_distance:
                continue

            err_d = by_distance[distance]
            err_next = by_distance[next_distance]
            suppression_factor = err_d / err_next

            suppression_records.append(
                {
                    "p_data": p_data,
                    "distance": distance,
                    "next_distance": next_distance,
                    "lambda": suppression_factor,
                }
            )

    return suppression_records


distances = [3, 5]
physical_error_rates = [0.005, 0.01]

results = run_parameter_sweep(
    distances,
    physical_error_rates,
    1000,
    0.2
)

for result in results:
    print(
        result["distance"],
        result["p_data"],
        result["status"],
        result["p_logical_total"]
    )

suppression = calculate_error_suppression(results)
for entry in suppression:
    print(entry)


def save_results(results, config, output_dir, suppression_results=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert NumPy values into standard Python values for JSON
    def convert_numpy(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert_numpy(value) for value in obj]
        return obj

    # 1. Save experiment results to CSV
    if results:
        results = convert_numpy(results)
        fieldnames = list(dict.fromkeys(
            key for entry in results for key in entry
        ))

        results_path = output_dir / "experiment_results.csv"

        with open(results_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    else:
        raise ValueError("No experiment results to save")

    # 2. Save error suppression results to CSV
    if suppression_results is not None:
        suppression_results = convert_numpy(suppression_results)

        if suppression_results:
            suppression_path = output_dir / "error_suppression.csv"

            with open(suppression_path, "w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(
                    file, fieldnames=suppression_results[0].keys()
                )
                writer.writeheader()
                writer.writerows(suppression_results)

    # 3. Save experiment configuration to JSON
    config_path = output_dir / "experiment_config.json"

    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(convert_numpy(config), file, indent=4)

    return output_dir

for r in results:
    print(
        r["p_data"],
        r["distance"],
        r["status"],
        r["p_logical_round"]
    )