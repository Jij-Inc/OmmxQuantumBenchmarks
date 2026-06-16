def parse_instance_xml(file_path: str) -> dict:
    import xml.etree.ElementTree as ET
    import numpy as np

    tree = ET.parse(file_path)
    root = tree.getroot()
    data = {}

    # =====================
    # Basic Parameters
    # =====================
    teams = root.find(".//Teams")
    slots = root.find(".//Slots")

    T = len(teams.findall("team"))
    S = len(slots.findall("slot"))

    data["T"] = T
    data["S"] = S
    data["S0"] = S - 1  # slots 1..S-1
    data["S1"] = S // 2  # first half
    data["S2"] = S - data["S1"]  # second half
    data["SZ"] = S - 1
    data["matches_per_slot"] = T // 2

    # =====================
    # Index Sets
    # =====================
    data["T_idx"] = np.arange(T)
    data["S_idx"] = np.arange(S)
    data["S0_idx"] = np.arange(1, S)
    data["S0_prev"] = np.arange(0, S - 1)
    data["S1_idx"] = np.arange(0, data["S1"])
    data["S2_idx"] = np.arange(data["S1"], S)
    data["SZ_idx"] = np.arange(0, S - 1)

    pairs_M = [(m, n) for m in range(T) for n in range(T) if m != n]
    pairs_MR = [(m, n) for m in range(T) for n in range(m + 1, T)]
    data["M_n"] = len(pairs_M)
    data["M_pair"] = np.array(pairs_M, dtype=int)
    data["MR_n"] = len(pairs_MR)
    data["MR_pair"] = np.array(pairs_MR, dtype=int)

    # =====================
    # CapacityConstraints → CA4
    # =====================
    ca4_list = root.findall(".//CA4")
    A_sets, B_sets, W_sets, U_caps = [], [], [], []
    for ca4 in ca4_list:
        A = (
            list(map(int, ca4.attrib["teams1"].split(";")))
            if ca4.attrib["teams1"]
            else []
        )
        B = (
            list(map(int, ca4.attrib["teams2"].split(";")))
            if ca4.attrib["teams2"]
            else []
        )
        W = (
            list(map(int, ca4.attrib["slots"].split(";")))
            if ca4.attrib["slots"]
            else []
        )
        U = int(ca4.attrib["max"])
        A_sets.append(A)
        B_sets.append(B)
        W_sets.append(W)
        U_caps.append(U)

    if A_sets:
        data["B_ca"] = len(A_sets)
        data["Amax"] = max(len(a) for a in A_sets)
        data["Bmax"] = max(len(b) for b in B_sets)
        data["Wmax"] = max(len(w) for w in W_sets)

        def pad_to(arrs, maxlen, fill=-1):
            return np.array(
                [row + [fill] * (maxlen - len(row)) for row in arrs], dtype=int
            )

        data["A_set"] = pad_to(A_sets, data["Amax"])
        data["A_len"] = np.array([len(a) for a in A_sets], dtype=int)
        data["B_set"] = pad_to(B_sets, data["Bmax"])
        data["B_len"] = np.array([len(b) for b in B_sets], dtype=int)
        data["W_set"] = pad_to(W_sets, data["Wmax"])
        data["W_len"] = np.array([len(w) for w in W_sets], dtype=int)
        data["U_cap"] = np.array(U_caps, dtype=int)
    else:
        data["B_ca"] = 0
        data["A_set"] = np.empty((0, 0), dtype=int)
        data["B_set"] = np.empty((0, 0), dtype=int)
        data["W_set"] = np.empty((0, 0), dtype=int)
        data["U_cap"] = np.array([], dtype=int)

    # =====================
    # GameConstraints → GA1 (lower bound)
    # =====================
    ga1_list = root.findall(".//GA1")
    if ga1_list:
        pairs_list, wins_list, bounds = [], [], []
        for ga in ga1_list:
            lb = int(ga.attrib["min"])
            teams1 = (
                list(map(int, ga.attrib["teams1"].split(";")))
                if ga.attrib.get("teams1")
                else []
            )
            teams2 = (
                list(map(int, ga.attrib["teams2"].split(";")))
                if ga.attrib.get("teams2")
                else []
            )
            slots = (
                list(map(int, ga.attrib["slots"].split(";")))
                if ga.attrib.get("slots")
                else []
            )

            # ✅ 跳過空集合的 GA1，避免 infeasible
            if not teams1 or not teams2 or not slots:
                continue

            pairs = list(zip(teams1, teams2))
            pairs_list.append(pairs)
            wins_list.append(slots)
            bounds.append(lb)

        if pairs_list:
            max_pairs = max(len(p) for p in pairs_list)
            max_wins = max(len(w) for w in wins_list)

            def pad_pairs(arrs, maxlen):
                return np.array(
                    [row + [(-1, -1)] * (maxlen - len(row)) for row in arrs], dtype=int
                )

            def pad_slots(arrs, maxlen):
                return np.array(
                    [row + [-1] * (maxlen - len(row)) for row in arrs], dtype=int
                )

            data["L_ga_n"] = len(pairs_list)
            data["L_pairsN"] = sum(len(p) for p in pairs_list)
            data["L_winsN"] = sum(len(w) for w in wins_list)
            data["L_pairs"] = pad_pairs(pairs_list, max_pairs)
            data["L_pair_len"] = np.array([len(p) for p in pairs_list], dtype=int)
            data["L_wins"] = pad_slots(wins_list, max_wins)
            data["L_win_len"] = np.array([len(w) for w in wins_list], dtype=int)
            data["L_bound"] = np.array(bounds, dtype=int)
        else:
            data["L_ga_n"] = 0
            data["L_pairs"] = np.empty((0, 0, 2), dtype=int)
            data["L_wins"] = np.empty((0, 0), dtype=int)
    else:
        data["L_ga_n"] = 0
        data["L_pairs"] = np.empty((0, 0, 2), dtype=int)
        data["L_wins"] = np.empty((0, 0), dtype=int)

    # Default UGA
    data["U_ga_n"] = 0
    data["U_pairs"] = np.empty((0, 0, 2), dtype=int)
    data["U_wins"] = np.empty((0, 0), dtype=int)

    # =====================
    # BreakConstraints → BR1 / BR2
    # =====================
    br1_list = root.findall(".//BR1")
    if br1_list:
        teams, wins, wlens, ubs = [], [], [], []
        for br in br1_list:
            t = int(br.attrib["teams"])
            ub = int(br.attrib["intp"])
            slots = (
                list(map(int, br.attrib["slots"].split(";")))
                if br.attrib["slots"]
                else []
            )
            teams.append(t)
            wins.append(slots)
            wlens.append(len(slots))
            ubs.append(ub)

        max_w = max(wlens)
        wins_padded = [row + [-1] * (max_w - len(row)) for row in wins]
        data["BR1_n"] = len(teams)
        data["BR1_wN"] = sum(wlens)
        data["BR1_teams"] = np.array(teams, dtype=int)
        data["BR1_wins"] = np.array(wins_padded, dtype=int)
        data["BR1_wlen"] = np.array(wlens, dtype=int)
        data["BR1_ub"] = np.array(ubs, dtype=int)
    else:
        data["BR1_n"] = 0
        data["BR1_teams"] = np.array([], dtype=int)
        data["BR1_wins"] = np.empty((0, 0), dtype=int)

    br2 = root.find(".//BR2")
    if br2 is not None:
        data["BR2_cap"] = int(br2.attrib["intp"])
    else:
        data["BR2_cap"] = 0

    return data
