import random
from env.logistics_env import LogisticsInstance
from planner.or_tools_planner import solve_vrptw
from reasoner.z3_validator import validate_solution_with_z3
from agents.perception import PerceptionNet, predict_symbol
from data.perception_synth import render_customer_image
import torch

def build_data_from_instance(inst: LogisticsInstance):
    return {
        'distance_matrix': inst.distance_matrix,
        'demands': inst.demands,
        'time_windows': inst.time_windows,
        'service_times': inst.service_times,
        'vehicle_count': inst.vehicle_count,
        'vehicle_capacity': inst.vehicle_capacity
    }

def pretty_print_solution(inst, solution):
    print("\n== OR-Tools solution ==")
    routes = solution.get('routes', [])
    times = solution.get('times', [])
    loads = solution.get('loads', [])
    for vid, route in enumerate(routes):
        if len(route) == 0:
            print(f"Vehicle {vid}: empty")
            continue
        entries = []
        tlist = times[vid] if vid < len(times) else [None]*len(route)
        llist = loads[vid] if vid < len(loads) else [None]*len(route)
        for i, node in enumerate(route):
            t = tlist[i] if i < len(tlist) else None
            l = llist[i] if i < len(llist) else None
            if node == 0:
                entries.append(f"Depot(t={t},load={l})")
            else:
                cust = inst.customers[node-1]
                entries.append(f"C{cust.id}(t={t},load={l},tw=({cust.tw_start},{cust.tw_end}))")
        print(f"Vehicle {vid}: " + " -> ".join(entries))
    print("Objective (approx distance):", solution.get('objective', 'N/A'))

def main():
    seed = 42
    inst = LogisticsInstance(n_customers=10, seed=seed, max_demand=3,
                             vehicle_count=3, vehicle_capacity=10,
                             time_window_slack=40, tightness=0.8)

    inst.summary()

    data = build_data_from_instance(inst)

    print("\nSolving VRPTW with OR-Tools (time_limit=60s)...")
    solution = solve_vrptw(data, time_limit=60)

    if solution is None:
        print("No solution found by OR-Tools.")
        return

    pretty_print_solution(inst, solution)


    model = PerceptionNet(in_ch=1, num_classes=4)
    model.load_state_dict(torch.load('models/perception_net_best.pth', map_location='cpu'))

    print("\nPerception agent predictions (customer -> predicted demand, confidence, probs):")
    for cust in inst.customers:
        pil = render_customer_image((cust.x, cust.y), (inst.depot.x, inst.depot.y),
                                    demand_label=cust.demand, size=16)
        pred, conf, probs = predict_symbol(model, pil)
        probs_str = ", ".join([f"{p:.2f}" for p in probs])
        print(f"Cust {cust.id}: predicted_class={pred}, confidence={conf:.2f}, probs=[{probs_str}]  (true demand={cust.demand})")



    print("\n=== DEBUG: RAW SOLUTION OBJECT ===")
    print(solution)
    print("===================================\n")

    print("Validating solution with Z3-style validator...")
    valid, details = validate_solution_with_z3(inst, solution)
    print("\nValidation result:", "VALID" if valid else "INVALID")
    if not valid:
        print("Violations / details:")
        if isinstance(details, list):
            for d in details:
                print(" -", d)
        else:
            print(details)
    else:
        print(details)

if __name__ == "__main__":
    main()
