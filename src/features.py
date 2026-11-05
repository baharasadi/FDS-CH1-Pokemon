from collections import Counter

import numpy as np
import pandas as pd

SIDES = ("p1", "p2")
OTHER = {"p1": "p2", "p2": "p1"}
STATS = ("base_hp", "base_atk", "base_def", "base_spa", "base_spd", "base_spe")
STATUSES = ("slp", "par", "tox", "psn", "frz", "brn")
EFFECTS = ("confusion", "leechseed", "substitute", "trapped", "protect", "perish", "encore", "attract")


def build_species_table(battles):
    """Types and base stats per species, taken from the team and lead details in the data."""
    table = {}
    for battle in battles:
        for pokemon in battle["p1_team_details"] + [battle["p2_lead_details"]]:
            if pokemon["name"] not in table:
                entry = {stat: pokemon[stat] for stat in STATS}
                entry["types"] = [t for t in pokemon["types"] if t != "notype"]
                entry["bst"] = sum(pokemon[stat] for stat in STATS)
                table[pokemon["name"]] = entry

    # a few rare P2 species only appear in the timeline, so we don't know their stats
    average = {stat: float(np.mean([e[stat] for e in table.values()])) for stat in STATS + ("bst",)}
    for battle in battles:
        for turn in battle["battle_timeline"]:
            name = turn["p2_pokemon_state"]["name"]
            if name not in table:
                table[name] = dict(average, types=[])
    return table


def build_vocab(battles, min_species=50, n_moves=40):
    """Species and moves common enough to get their own column (train data only)."""
    species = Counter()
    moves = Counter()
    for battle in battles:
        seen = {t[f"{side}_pokemon_state"]["name"] for t in battle["battle_timeline"] for side in SIDES}
        species.update(seen)
        for turn in battle["battle_timeline"]:
            for side in SIDES:
                move = turn[f"{side}_move_details"]
                if move:
                    moves[move["name"]] += 1
    return {
        "species": sorted(name for name, count in species.items() if count >= min_species),
        "moves": sorted(name for name, _ in moves.most_common(n_moves)),
    }


def type_multiplier(attack_type, defending_types, chart):
    result = 1.0
    for defending_type in defending_types:
        # the chart only lists non-neutral pairs
        result *= chart.get((attack_type, defending_type), 1.0)
    return result


def best_stab_multiplier(attacker, defender, species, chart):
    attacker_types = species[attacker]["types"]
    defender_types = species[defender]["types"]
    if not attacker_types:
        return 1.0
    return max(type_multiplier(t, defender_types, chart) for t in attacker_types)


def add_differences(row, names):
    for name in names:
        row[f"d_{name}"] = row[f"p1_{name}"] - row[f"p2_{name}"]


def last_states(turns, side):
    # dict keeps the order of first appearance, values are the latest state
    last = {}
    for turn in turns:
        state = turn[f"{side}_pokemon_state"]
        last[state["name"]] = state
    return last


def count_likely_kos(turns, side):
    """
    Fainted P2 Pokemon are rarely recorded in the timeline. A Pokemon that leaves on low HP
    while being attacked, is replaced by a fresh one and never comes back probably fainted.
    Checked on the data: about 80% of these never return, against 41% of all switch-outs.
    """
    names = [t[f"{side}_pokemon_state"]["name"] for t in turns]
    hp = [t[f"{side}_pokemon_state"]["hp_pct"] for t in turns]
    count = 0
    for i in range(1, len(turns)):
        if names[i] == names[i - 1] or hp[i - 1] == 0 or hp[i - 1] >= 0.5 or hp[i] < 0.87:
            continue
        move = turns[i][f"{OTHER[side]}_move_details"]
        attacked = bool(move) and move["category"] != "status"
        if attacked and names[i - 1] not in names[i:]:
            count += 1
    return count


# Feature groups. Each one takes (battle, species, chart, vocab) and returns a dict.


def base_features(battle, species, chart, vocab):
    """General HP, status, effect and move power summaries, plus p1 minus p2 differences."""
    turns = battle["battle_timeline"]
    team = battle["p1_team_details"]
    lead = battle["p2_lead_details"]
    row = {
        "p1_mean_hp": np.mean([p["base_hp"] for p in team]),
        "p1_mean_spe": np.mean([p["base_spe"] for p in team]),
        "p1_mean_atk": np.mean([p["base_atk"] for p in team]),
        "p1_mean_def": np.mean([p["base_def"] for p in team]),
        "p2_lead_hp": lead["base_hp"],
        "p2_lead_spe": lead["base_spe"],
    }
    hp_series = {side: [t[f"{side}_pokemon_state"]["hp_pct"] for t in turns] for side in SIDES}
    row["last_hp_difference"] = hp_series["p1"][-1] - hp_series["p2"][-1]

    paired = []
    for side in SIDES:
        hp = hp_series[side]
        states = [t[f"{side}_pokemon_state"] for t in turns]
        row[f"{side}_timeline_mean_hp"] = np.mean(hp)
        row[f"{side}_timeline_min_hp"] = np.min(hp)
        row[f"{side}_hp0_count"] = len({s["name"] for s in states if s["hp_pct"] == 0})

        last_hp = {}
        appearances = Counter()
        recovered = 0.0
        for state in states:
            if state["name"] in last_hp:
                recovered += max(state["hp_pct"] - last_hp[state["name"]], 0.0)
            last_hp[state["name"]] = state["hp_pct"]
            appearances[state["name"]] += 1
        row[f"{side}_alive_appearances"] = sum(appearances[n] for n, h in last_hp.items() if h > 0)
        row[f"{side}_last_mean_hp"] = np.mean(list(last_hp.values()))
        row[f"{side}_unique_pokemon"] = len(last_hp)
        row[f"{side}_hp_recovered"] = recovered

        for status in STATUSES:
            row[f"{side}_status_{status}"] = sum(s["status"] == status for s in states)
        for effect in EFFECTS:
            row[f"{side}_effect_{effect}"] = sum(effect in s["effects"] for s in states)

        powers = [t[f"{side}_move_details"]["base_power"] for t in turns if t[f"{side}_move_details"]]
        row[f"{side}_mean_power"] = np.mean(powers) if powers else 0.0

    for name in list(row):
        if name.startswith("p1_") and f"p2_{name[3:]}" in row:
            paired.append(name[3:])
    add_differences(row, paired)
    return row


def board_features(battle, species, chart, vocab):
    """What is left on each side after turn 30. Pokemon not seen yet count as full HP."""
    turns = battle["battle_timeline"]
    row = {}
    for side in SIDES:
        last = last_states(turns, side)
        active = turns[-1][f"{side}_pokemon_state"]["name"]
        alive = {name: s for name, s in last.items() if s["hp_pct"] > 0}
        benched = [s for name, s in alive.items() if name != active]
        unseen = max(6 - len(last), 0)
        row[f"{side}_fainted"] = len(last) - len(alive)
        row[f"{side}_unseen"] = unseen
        row[f"{side}_total_hp"] = sum(s["hp_pct"] for s in alive.values()) + unseen
        row[f"{side}_statused_alive"] = sum(s["status"] not in ("nostatus", "fnt") for s in alive.values())
        row[f"{side}_asleep_alive"] = sum(s["status"] == "slp" for s in alive.values())
        row[f"{side}_low_alive"] = sum(s["hp_pct"] < 0.35 for s in alive.values())
        row[f"{side}_benched_lt25"] = sum(s["hp_pct"] < 0.25 for s in benched)
        row[f"{side}_benched_lt50"] = sum(s["hp_pct"] < 0.50 for s in benched)
        row[f"{side}_alive_hp_mean"] = np.mean([s["hp_pct"] for s in alive.values()]) if alive else 0.0
        row[f"{side}_healthy_alive"] = sum(s["hp_pct"] > 0.7 and s["status"] == "nostatus" for s in alive.values())
        row[f"{side}_weighted_hp"] = sum(s["hp_pct"] * species[n]["bst"] for n, s in alive.items()) / 100
        row[f"{side}_active_hp"] = turns[-1][f"{side}_pokemon_state"]["hp_pct"]
    add_differences(row, [
        "fainted", "unseen", "total_hp", "statused_alive", "asleep_alive", "low_alive",
        "benched_lt25", "benched_lt50", "alive_hp_mean", "healthy_alive", "weighted_hp", "active_hp",
    ])
    return row


def ko_features(battle, species, chart, vocab):
    """Estimate of the real faint count, since P2 faints are under-recorded."""
    turns = battle["battle_timeline"]
    row = {}
    for side in SIDES:
        last = last_states(turns, side)
        recorded = sum(s["hp_pct"] == 0 for s in last.values())
        likely = count_likely_kos(turns, side)
        row[f"{side}_likely_kos"] = likely
        row[f"{side}_estimated_faints"] = recorded + likely
    add_differences(row, ["likely_kos", "estimated_faints"])
    return row


def trend_features(battle, species, chart, vocab):
    """Board state at earlier checkpoints, to tell who is gaining ground."""
    turns = battle["battle_timeline"]
    row = {}
    for checkpoint in (10, 20, 30):
        part = turns[:checkpoint]
        for side in SIDES:
            last = last_states(part, side)
            alive_hp = sum(s["hp_pct"] for s in last.values())
            row[f"{side}_hp_t{checkpoint}"] = alive_hp + max(6 - len(last), 0)
        row[f"d_hp_t{checkpoint}"] = row[f"p1_hp_t{checkpoint}"] - row[f"p2_hp_t{checkpoint}"]
    row["d_hp_gain_first"] = row["d_hp_t20"] - row["d_hp_t10"]
    row["d_hp_gain_last"] = row["d_hp_t30"] - row["d_hp_t20"]

    for side in SIDES:
        damage = {"early": 0.0, "late": 0.0}
        previous = {}
        for i, turn in enumerate(turns):
            state = turn[f"{side}_pokemon_state"]
            if state["name"] in previous:
                loss = previous[state["name"]] - state["hp_pct"]
                if loss > 0:
                    damage["early" if i < 15 else "late"] += loss
            previous[state["name"]] = state["hp_pct"]
        row[f"{side}_damage_early"] = damage["early"]
        row[f"{side}_damage_late"] = damage["late"]
    add_differences(row, ["damage_early", "damage_late"])
    return row


def move_features(battle, species, chart, vocab):
    """How good the chosen moves were against the Pokemon they were aimed at."""
    turns = battle["battle_timeline"]
    row = {}
    for side in SIDES:
        other = OTHER[side]
        damage_scores = []
        super_hits = resisted = immune = stab = status_moves = no_move = 0
        for turn in turns:
            move = turn[f"{side}_move_details"]
            if not move:
                no_move += 1
                continue
            if move["category"] == "status" or move["base_power"] == 0:
                status_moves += 1
                continue
            attacker = species[turn[f"{side}_pokemon_state"]["name"]]
            defender = species[turn[f"{other}_pokemon_state"]["name"]]
            multiplier = type_multiplier(move["type"], defender["types"], chart)
            has_stab = move["type"] in attacker["types"]
            if move["category"] == "physical":
                ratio = attacker["base_atk"] / defender["base_def"]
            else:
                ratio = attacker["base_spa"] / defender["base_spd"]
            damage_scores.append(move["base_power"] * multiplier * ratio * (1.5 if has_stab else 1.0))
            super_hits += multiplier > 1
            resisted += 0 < multiplier < 1
            immune += multiplier == 0
            stab += has_stab
        row[f"{side}_damage_score_mean"] = np.mean(damage_scores) if damage_scores else 0.0
        row[f"{side}_damage_score_sum"] = np.sum(damage_scores)
        row[f"{side}_super_hits"] = super_hits
        row[f"{side}_resisted_hits"] = resisted
        row[f"{side}_immune_hits"] = immune
        row[f"{side}_stab_hits"] = stab
        row[f"{side}_status_moves"] = status_moves
        row[f"{side}_no_move_turns"] = no_move
    add_differences(row, [
        "damage_score_mean", "damage_score_sum", "super_hits", "resisted_hits",
        "immune_hits", "stab_hits", "status_moves", "no_move_turns",
    ])
    return row


def matchup_features(battle, species, chart, vocab):
    """Type and speed matchup between the Pokemon each side still has."""
    turns = battle["battle_timeline"]
    p1_alive = [n for n, s in last_states(turns, "p1").items() if s["hp_pct"] > 0]
    p2_alive = [n for n, s in last_states(turns, "p2").items() if s["hp_pct"] > 0]
    # P1's team is fully known, so unseen P1 Pokemon count as well
    seen = set(last_states(turns, "p1"))
    p1_alive += [p["name"] for p in battle["p1_team_details"] if p["name"] not in seen]

    row = {}
    if p1_alive and p2_alive:
        row["p1_type_edge"] = np.mean([best_stab_multiplier(a, b, species, chart) for a in p1_alive for b in p2_alive])
        row["p2_type_edge"] = np.mean([best_stab_multiplier(b, a, species, chart) for a in p1_alive for b in p2_alive])
    else:
        row["p1_type_edge"] = row["p2_type_edge"] = 1.0
    row["p1_alive_speed"] = max([species[n]["base_spe"] for n in p1_alive], default=0)
    row["p2_alive_speed"] = max([species[n]["base_spe"] for n in p2_alive], default=0)
    row["p1_alive_offense"] = max([max(species[n]["base_atk"], species[n]["base_spa"]) for n in p1_alive], default=0)
    row["p2_alive_offense"] = max([max(species[n]["base_atk"], species[n]["base_spa"]) for n in p2_alive], default=0)
    add_differences(row, ["type_edge", "alive_speed", "alive_offense"])

    last = turns[-1]
    p1_active = last["p1_pokemon_state"]["name"]
    p2_active = last["p2_pokemon_state"]["name"]
    row["active_type_edge"] = (
        best_stab_multiplier(p1_active, p2_active, species, chart)
        - best_stab_multiplier(p2_active, p1_active, species, chart)
    )
    row["active_speed_diff"] = species[p1_active]["base_spe"] - species[p2_active]["base_spe"]
    return row


def boost_features(battle, species, chart, vocab):
    turns = battle["battle_timeline"]
    row = {}
    for side in SIDES:
        last = turns[-1][f"{side}_pokemon_state"]["boosts"]
        row[f"{side}_last_boost_sum"] = sum(last.values())
        row[f"{side}_mean_boost_sum"] = np.mean([sum(t[f"{side}_pokemon_state"]["boosts"].values()) for t in turns])
    add_differences(row, ["last_boost_sum", "mean_boost_sum"])
    return row


def team_features(battle, species, chart, vocab):
    team = battle["p1_team_details"]
    lead = battle["p2_lead_details"]
    row = {
        "team_bst_mean": np.mean([species[p["name"]]["bst"] for p in team]),
        "team_spe_max": max(p["base_spe"] for p in team),
        "team_leftovers": sum(p.get("item") == "leftovers" for p in team),
        "team_distinct_types": len({t for p in team for t in species[p["name"]]["types"]}),
        "lead_bst_diff": species[team[0]["name"]]["bst"] - species[lead["name"]]["bst"],
        "lead_spe_diff": team[0]["base_spe"] - lead["base_spe"],
    }
    return row


def species_features(battle, species, chart, vocab):
    turns = battle["battle_timeline"]
    p1_team = {p["name"] for p in battle["p1_team_details"]}
    p2_seen = set(last_states(turns, "p2"))
    row = {}
    for name in vocab["species"]:
        row[f"p1_has_{name}"] = int(name in p1_team)
        row[f"p2_has_{name}"] = int(name in p2_seen)
    return row


def move_bag_features(battle, species, chart, vocab):
    row = {}
    for side in SIDES:
        counts = Counter(
            t[f"{side}_move_details"]["name"]
            for t in battle["battle_timeline"]
            if t[f"{side}_move_details"]
        )
        for name in vocab["moves"]:
            row[f"{side}_used_{name}"] = counts[name]
    return row


def vanish_features(battle, species, chart, vocab):
    """Pokemon that left the field and stayed away. The longer and the lower the HP, the more likely it fainted."""
    turns = battle["battle_timeline"]
    row = {}
    names = ["gone5_low", "gone10_low", "gone10_any", "gone15_any", "ko_score", "estimated_alive", "estimated_hp"]
    for side in SIDES:
        last_seen = {}
        for i, turn in enumerate(turns):
            state = turn[f"{side}_pokemon_state"]
            last_seen[state["name"]] = (i, state["hp_pct"])
        active = turns[-1][f"{side}_pokemon_state"]["name"]
        gone = [
            (len(turns) - 1 - i, hp)
            for name, (i, hp) in last_seen.items()
            if name != active and hp > 0
        ]
        probable_kos = sum(hp < 0.5 and away >= 5 for away, hp in gone)
        fainted = sum(hp == 0 for _, hp in last_seen.values())
        seen_hp = sum(hp for _, hp in last_seen.values())
        lost_hp = sum(hp for away, hp in gone if hp < 0.5 and away >= 5)
        unseen = max(6 - len(last_seen), 0)
        row[f"{side}_gone5_low"] = probable_kos
        row[f"{side}_gone10_low"] = sum(hp < 0.5 and away >= 10 for away, hp in gone)
        row[f"{side}_gone10_any"] = sum(away >= 10 for away, _ in gone)
        row[f"{side}_gone15_any"] = sum(away >= 15 for away, _ in gone)
        row[f"{side}_ko_score"] = sum((1 - hp) * min(away, 15) / 15 for away, hp in gone)
        row[f"{side}_estimated_alive"] = 6 - fainted - probable_kos
        row[f"{side}_estimated_hp"] = seen_hp - lost_hp + unseen
    add_differences(row, names)
    return row


def status_features(battle, species, chart, vocab):
    """Which status problems each side has, how recent they are and how much HP they affect."""
    turns = battle["battle_timeline"]
    weights = {"nostatus": 1.0, "par": 0.8, "psn": 0.8, "brn": 0.8, "tox": 0.7, "slp": 0.5, "frz": 0.3}
    row = {}
    names = ["recent_status", "recent_slp", "alive_par", "alive_poison", "alive_frz", "inflicted", "cured", "useful_hp"]
    for side in SIDES:
        states = [t[f"{side}_pokemon_state"] for t in turns]
        alive = [s for s in last_states(turns, side).values() if s["hp_pct"] > 0]
        inflicted = cured = 0
        previous = {}
        for state in states:
            before = previous.get(state["name"], "nostatus")
            inflicted += before == "nostatus" and state["status"] not in ("nostatus", "fnt")
            cured += before not in ("nostatus", "fnt") and state["status"] == "nostatus"
            previous[state["name"]] = state["status"]
        row[f"{side}_recent_status"] = sum(s["status"] not in ("nostatus", "fnt") for s in states[-10:])
        row[f"{side}_recent_slp"] = sum(s["status"] == "slp" for s in states[-10:])
        row[f"{side}_alive_par"] = sum(s["status"] == "par" for s in alive)
        row[f"{side}_alive_poison"] = sum(s["status"] in ("psn", "tox") for s in alive)
        row[f"{side}_alive_frz"] = sum(s["status"] == "frz" for s in alive)
        row[f"{side}_inflicted"] = inflicted
        row[f"{side}_cured"] = cured
        row[f"{side}_useful_hp"] = sum(s["hp_pct"] * weights[s["status"]] for s in alive)
    add_differences(row, names)
    row["p1_active_asleep"] = int(turns[-1]["p1_pokemon_state"]["status"] == "slp")
    row["p2_active_asleep"] = int(turns[-1]["p2_pokemon_state"]["status"] == "slp")
    return row


def hazard_features(battle, species, chart, vocab):
    """Spikes set by each side, and whether they are still on the field at the end."""
    turns = battle["battle_timeline"]
    set_by = {"p1": False, "p2": False}
    turns_up = {"p1": 0, "p2": 0}
    uses = {"p1": 0, "p2": 0}
    spins = {"p1": 0, "p2": 0}
    for turn in turns:
        for side in SIDES:
            move = turn[f"{side}_move_details"]
            name = move["name"] if move else None
            if name == "spikes":
                set_by[side] = True
                uses[side] += 1
            elif name == "rapidspin":
                set_by[OTHER[side]] = False
                spins[side] += 1
        for side in SIDES:
            turns_up[side] += set_by[side]
    row = {}
    for side in SIDES:
        row[f"{side}_spikes_up"] = int(set_by[side])
        row[f"{side}_spikes_turns"] = turns_up[side]
        row[f"{side}_spikes_used"] = uses[side]
        row[f"{side}_spins"] = spins[side]
    add_differences(row, ["spikes_up", "spikes_turns", "spikes_used", "spins"])
    return row


def momentum_features(battle, species, chart, vocab):
    """How the HP lead moved during the 30 turns, and how recent the losses are."""
    turns = battle["battle_timeline"]
    last_hp = {"p1": {}, "p2": {}}
    lead = []
    for turn in turns:
        totals = {}
        for side in SIDES:
            state = turn[f"{side}_pokemon_state"]
            last_hp[side][state["name"]] = state["hp_pct"]
            totals[side] = sum(last_hp[side].values()) + max(6 - len(last_hp[side]), 0)
        lead.append(totals["p1"] - totals["p2"])
    lead = np.array(lead)
    positions = np.arange(len(lead))
    row = {
        "lead_slope": np.polyfit(positions, lead, 1)[0],
        "lead_slope_last10": np.polyfit(positions[-10:], lead[-10:], 1)[0],
        "lead_mean_last5": lead[-5:].mean(),
        "lead_mean_last10": lead[-10:].mean(),
        "turns_ahead": (lead > 0).sum(),
        "turns_ahead_last10": (lead[-10:] > 0).sum(),
        "lead_best": lead.max(),
        "lead_worst": lead.min(),
    }
    for side in SIDES:
        states = [t[f"{side}_pokemon_state"] for t in turns]
        faint_rows = [i for i, s in enumerate(states) if s["hp_pct"] == 0]
        row[f"{side}_turns_since_faint"] = len(turns) - 1 - faint_rows[-1] if faint_rows else len(turns)
        row[f"{side}_faints_last10"] = sum(i >= len(turns) - 10 for i in faint_rows)
        row[f"{side}_switches_last10"] = sum(
            a["name"] != b["name"] for a, b in zip(states[-11:], states[-10:])
        )
    add_differences(row, ["turns_since_faint", "faints_last10", "switches_last10"])
    return row


FEATURE_GROUPS = {
    "base": base_features,
    "board": board_features,
    "ko": ko_features,
    "trend": trend_features,
    "moves": move_features,
    "matchup": matchup_features,
    "boosts": boost_features,
    "team": team_features,
    "species": species_features,
    "move_bag": move_bag_features,
    "vanish": vanish_features,
    "status": status_features,
    "hazards": hazard_features,
    "momentum": momentum_features,
}


def build_features(battles, species, chart, vocab, groups=None):
    groups = groups or list(FEATURE_GROUPS)
    records = []
    for battle in battles:
        row = {}
        for group in groups:
            for name, value in FEATURE_GROUPS[group](battle, species, chart, vocab).items():
                row[f"{group}__{name}"] = value
        records.append(row)
    return pd.DataFrame(records).astype(float)


def select_columns(X, groups):
    return [c for c in X.columns if c.split("__")[0] in groups]
