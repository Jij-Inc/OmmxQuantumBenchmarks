import jijmodeling as jm


def build_drr_model_exact():
    P = jm.Problem("2RR Scheduling")

    # -------------------------
    # Sizes (scalars) = ZPL params and set sizes
    # -------------------------
    T = jm.Placeholder("T")  # |T| (ZPL: param teams)
    S = jm.Placeholder("S")  # |S| (ZPL: param slots)
    S0 = jm.Placeholder("S0")  # |S0| = S-1 (ZPL: S0 := {1..S-1})
    S1 = jm.Placeholder("S1")  # |S1| = S/2 (ZPL: S1 := {0..S/2-1})
    S2 = jm.Placeholder("S2")  # |S2| = S/2 (ZPL: S2 := {S/2..S-1})
    SZ = jm.Placeholder("SZ")  # |SZ| = S-1 (ZPL: SZ := {0..S-2})
    matches_per_slot = jm.Placeholder(
        "matches_per_slot"
    )  # ZPL: param matches_per_slot = |T|/2

    # -------------------------
    # Explicit index lists for sets (mirror ZPL set literals)
    # -------------------------
    T_idx = jm.Placeholder("T_idx", ndim=1)  # [0..T-1]
    S_idx = jm.Placeholder("S_idx", ndim=1)  # [0..S-1]
    S0_idx = jm.Placeholder("S0_idx", ndim=1)  # [1..S-1]  (used by br_count and br1)
    S1_idx = jm.Placeholder("S1_idx", ndim=1)  # [0..S/2 - 1] (used by c4)
    S2_idx = jm.Placeholder(
        "S2_idx", ndim=1
    )  # [S/2..S-1]    (not strictly needed but kept for completeness)
    SZ_idx = jm.Placeholder("SZ_idx", ndim=1)  # [0..S-2]      (optional)

    # Pairs for M (m!=n) and MR (m<n), mirroring ZPL's "set M" and "set MR"
    M_n = jm.Placeholder("M_n")
    M_pair = jm.Placeholder("M_pair", ndim=2)  # shape (M_n, 2) rows <m,n> with m!=n
    MR_n = jm.Placeholder("MR_n")
    MR_pair = jm.Placeholder("MR_pair", ndim=2)  # shape (MR_n, 2) rows <m,n> with m<n

    # -------------------------
    # Decision variables (ZPL: var x[M*S], bh[T*S0], ba[T*S0])
    # -------------------------
    x = jm.BinaryVar(
        "x", shape=(T, S, T)
    )  # we'll index as x[m, s, n] to make "sum over n" & "sum over m" natural
    # it is equivalent to x[T,T,S]; just a dimension order choice.
    bh = jm.BinaryVar("bh", shape=(T, S))  # used only at s in S0_idx
    ba = jm.BinaryVar("ba", shape=(T, S))

    # -------------------------
    # Elements (loop indices)
    # -------------------------
    m = jm.Element("m", belong_to=T)
    n = jm.Element("n", belong_to=T)
    s = jm.Element("s", belong_to=S)
    ss0 = jm.Element("ss0", belong_to=S0)  # position in S0_idx
    q1 = jm.Element("q1", belong_to=S1)  # position in S1_idx
    rr = jm.Element("rr", belong_to=MR_n)  # row index in MR_pair
    mm = jm.Element("mm", belong_to=M_n)  # row index in M_pair
    t = jm.Element("t", belong_to=T)
    a = jm.Element("a", belong_to=T)  # sum over opponents a != t
    h = jm.Element("h", belong_to=T)  # sum over opponents h != t

    # Helper accessors (like dereferencing <m,n> in ZPL sets)
    def M_host(mm_):
        return M_pair[mm_, 0]

    def M_away(mm_):
        return M_pair[mm_, 1]

    def MR_left(rr_):
        return MR_pair[rr_, 0]

    def MR_right(rr_):
        return MR_pair[rr_, 1]

    # -------------------------
    # c1: For each ordered pair (m,n) in M, play exactly once at m's home
    #     ZPL: subto c1: forall <m,n> in M : sum <s> in S : x[m,n,s] == 1;
    # -------------------------
    P += jm.Constraint(
        "c1_each_pair_home_once",
        jm.sum(s, x[M_host(mm), s, M_away(mm)]) == 1,
        forall=[mm],
    )

    # -------------------------
    # c2: Per slot, exactly |T|/2 matches
    #     ZPL: subto c2: forall <s> in S: sum <m,n> in M : x[m,n,s] == matches_per_slot;
    # -------------------------
    P += jm.Constraint(
        "c2_matches_per_slot",
        jm.sum(mm, x[M_host(mm), s, M_away(mm)]) == matches_per_slot,
        forall=[s],
    )

    # -------------------------
    # c3: Per team & slot, exactly one game (home or away)
    #     ZPL: subto c3: forall <s> in S, forall <t> in T:
    #                    sum <t,n> in M : x[t,n,s] + sum <m,t> in M : x[m,t,s] == 1;
    #     We write it as sums over a!=t and h!=t.
    # -------------------------
    P += jm.Constraint(
        "c3_one_game_per_team_per_slot",
        jm.sum([(a, a != t)], x[t, s, a]) + jm.sum([(h, h != t)], x[h, s, t]) == 1,
        forall=[t, s],
    )

    # -------------------------
    # br_count: Break activation constraints (exactly as in your ZPL)
    #     ZPL:
    #       sum_a x[t,a,s-1] + sum_a x[t,a,s] - 1 <= bh[t,s]
    #       sum_h x[h,t,s-1] + sum_h x[h,t,s] - 1 <= ba[t,s]
    #     We pass s via S0_idx[ss0] and precompute s-1 via S0_prev[ss0].
    # -------------------------
    S0_prev = jm.Placeholder(
        "S0_prev", ndim=1
    )  # same length as S0_idx; S0_prev[pos] = S0_idx[pos] - 1

    P += jm.Constraint(
        "br_count_home",
        jm.sum([(a, a != t)], x[t, S0_prev[ss0], a])
        + jm.sum([(a, a != t)], x[t, S0_idx[ss0], a])
        - 1
        <= bh[t, S0_idx[ss0]],
        forall=[t, ss0],
    )

    P += jm.Constraint(
        "br_count_away",
        jm.sum([(h, h != t)], x[h, S0_prev[ss0], t])
        + jm.sum([(h, h != t)], x[h, S0_idx[ss0], t])
        - 1
        <= ba[t, S0_idx[ss0]],
        forall=[t, ss0],
    )

    # -------------------------
    # c4: Phased double round-robin (exactly one leg in first half S1)
    #     ZPL: subto c4: forall <m,n> in MR : sum <s> in S1 : (x[m,n,s] + x[n,m,s]) == 1;
    # -------------------------
    P += jm.Constraint(
        "c4_phased",
        jm.sum(
            q1,
            x[MR_left(rr), S1_idx[q1], MR_right(rr)]
            + x[MR_right(rr), S1_idx[q1], MR_left(rr)],
        )
        == 1,
        forall=[rr],
    )

    # -------------------------
    # ca4_*: Cartesian blocks with upper bounds
    #     ZPL example: sum <m,n,s> in A * B * W with m != n : x[m,n,s] <= U;
    #     We support many such lines as "blocks".
    # -------------------------
    B_ca = jm.Placeholder("B_ca")  # number of ca4 lines
    Amax = jm.Placeholder("Amax")  # max |A| across lines
    Bmax = jm.Placeholder("Bmax")  # max |B| across lines
    Wmax = jm.Placeholder("Wmax")  # max |W| across lines
    A_set = jm.Placeholder("A_set", ndim=2)  # (B_ca, Amax) team ids (pad if needed)
    B_set = jm.Placeholder("B_set", ndim=2)  # (B_ca, Bmax)
    W_set = jm.Placeholder("W_set", ndim=2)  # (B_ca, Wmax) slot ids
    A_len = jm.Placeholder("A_len", ndim=1)  # (B_ca,)
    B_len = jm.Placeholder("B_len", ndim=1)  # (B_ca,)
    W_len = jm.Placeholder("W_len", ndim=1)  # (B_ca,)
    U_cap = jm.Placeholder("U_cap", ndim=1)  # (B_ca,)

    b = jm.Element("b", belong_to=B_ca)
    ia = jm.Element("ia", belong_to=Amax)
    ib = jm.Element("ib", belong_to=Bmax)
    iw = jm.Element("iw", belong_to=Wmax)

    mA = A_set[b, ia]
    nB = B_set[b, ib]
    sW = W_set[b, iw]

    P += jm.Constraint(
        "ca4_block_caps",
        jm.sum(
            [(ia, ia < A_len[b])],
            jm.sum(
                [(ib, (ib < B_len[b]) & (nB != mA))],
                jm.sum([(iw, iw < W_len[b])], x[mA, sW, nB]),
            ),
        )
        <= U_cap[b],
        forall=[b],
    )

    # -------------------------
    # ga1_*: Upper and lower bounds over selected (host,away) pairs and slots
    #     ZPL has both <= and >= versions; we keep two lists to mirror them exactly.
    # -------------------------

    # Upper-bound list: sum_{(i,j) in P_u} sum_{s in W_u} x[i,s,j] <= U
    U_ga_n = jm.Placeholder("U_ga_n")
    U_pairsN = jm.Placeholder("U_pairsN")
    U_winsN = jm.Placeholder("U_winsN")
    U_pairs = jm.Placeholder("U_pairs", ndim=3)  # (U_ga_n, U_pairsN, 2)
    U_pair_len = jm.Placeholder("U_pair_len", ndim=1)
    U_wins = jm.Placeholder("U_wins", ndim=2)  # (U_ga_n, U_winsN)
    U_win_len = jm.Placeholder("U_win_len", ndim=1)
    U_bound = jm.Placeholder("U_bound", ndim=1)

    gu = jm.Element("gu", belong_to=U_ga_n)
    up = jm.Element("up", belong_to=U_pairsN)
    uw = jm.Element("uw", belong_to=U_winsN)

    ih_u = U_pairs[gu, up, 0]
    ja_u = U_pairs[gu, up, 1]
    sw_u = U_wins[gu, uw]

    P += jm.Constraint(
        "ga1_upper_blocks",
        jm.sum(
            [(uw, uw < U_win_len[gu])],
            jm.sum([(up, up < U_pair_len[gu])], x[ih_u, sw_u, ja_u]),
        )
        <= U_bound[gu],
        forall=[gu],
    )

    # Lower-bound list: sum_{(i,j) in P_l} sum_{s in W_l} x[i,s,j] >= L
    L_ga_n = jm.Placeholder("L_ga_n")
    L_pairsN = jm.Placeholder("L_pairsN")
    L_winsN = jm.Placeholder("L_winsN")
    L_pairs = jm.Placeholder("L_pairs", ndim=3)  # (L_ga_n, L_pairsN, 2)
    L_pair_len = jm.Placeholder("L_pair_len", ndim=1)
    L_wins = jm.Placeholder("L_wins", ndim=2)
    L_win_len = jm.Placeholder("L_win_len", ndim=1)
    L_bound = jm.Placeholder("L_bound", ndim=1)

    gl = jm.Element("gl", belong_to=L_ga_n)
    lp = jm.Element("lp", belong_to=L_pairsN)
    lw = jm.Element("lw", belong_to=L_winsN)

    ih_l = L_pairs[gl, lp, 0]
    ja_l = L_pairs[gl, lp, 1]
    sw_l = L_wins[gl, lw]

    P += jm.Constraint(
        "ga1_lower_blocks",
        jm.sum(
            [(lw, lw < L_win_len[gl])],
            jm.sum([(lp, lp < L_pair_len[gl])], x[ih_l, sw_l, ja_l]),
        )
        >= L_bound[gl],
        forall=[gl],
    )

    # -------------------------
    # br1_*: Per-team break caps over specified slot subsets
    #     ZPL: forall <t> in {..}: sum <s> in {...}: (bh[t,s] + ba[t,s]) <= UB;
    # -------------------------
    BR1_n = jm.Placeholder("BR1_n")
    BR1_wN = jm.Placeholder("BR1_wN")
    BR1_teams = jm.Placeholder(
        "BR1_teams", ndim=1
    )  # length BR1_n; each entry is a team id
    BR1_wins = jm.Placeholder("BR1_wins", ndim=2)  # (BR1_n, BR1_wN) slot ids
    BR1_wlen = jm.Placeholder("BR1_wlen", ndim=1)  # per-line actual length
    BR1_ub = jm.Placeholder("BR1_ub", ndim=1)  # per-line UB

    gb = jm.Element("gb", belong_to=BR1_n)
    bw = jm.Element("bw", belong_to=BR1_wN)
    tb = BR1_teams[gb]
    sw_b = BR1_wins[gb, bw]

    P += jm.Constraint(
        "br1_team_block_caps",
        jm.sum([(bw, bw < BR1_wlen[gb])], bh[tb, sw_b] + ba[tb, sw_b]) <= BR1_ub[gb],
        forall=[gb],
    )

    # -------------------------
    # br2_1: Global break cap
    #     ZPL: sum <t,s> in T * S0 : (bh[t,s] + ba[t,s]) <= 26;   (26 in your file)
    # -------------------------
    BR2_cap = jm.Placeholder("BR2_cap")
    P += jm.Constraint(
        "br2_global_cap",
        jm.sum(t, jm.sum(ss0, bh[t, S0_idx[ss0]] + ba[t, S0_idx[ss0]])) <= BR2_cap,
    )

    # -------------------------
    # Objective: feasibility (constant 0)
    # -------------------------
    P += jm.sum(t, 0)

    return P
