"""Cleaning pipeline for the Mario Kart 8 track dataset.

Produces data/mk8_tracks_clean.csv from data/mk8_tracks.csv.

The raw CSV was collected in two batches with incompatible encodings
(see README "Dataset" section for the full writeup). This script makes one
documented, reproducible decision per problem instead of silently coercing
values. Run with: python src/clean.py
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "mk8_tracks.csv"
CLEAN_CSV = ROOT / "data" / "mk8_tracks_clean.csv"
STAT_CHART_XLSX = ROOT / "reference" / "mk8_stat_chart_luigi_fan2.xlsx"

# ---------------------------------------------------------------------------
# 1. Recompute CBC_* targets from the stat chart instead of trusting the
#    hand-entered values (identical builds had different stats in different
#    rows in the raw CSV -- this is what fixes that).
# ---------------------------------------------------------------------------

# Deluxe-only parts not covered by Luigi_Fan2's (Wii U) chart. Rows that use
# any of these get flag_target_unverified=1 and keep their original CBC_*
# values untouched rather than a guessed recomputation.
UNCOVERED_PARTS = {"Birdo (Light Blue)", "Funky Kong", "Kamek", "Peachette", "Inkstriker"}

STAT_COLUMNS = [
    "speed_land", "speed_antigrav", "speed_water", "speed_glide",
    "accel", "weight",
    "handling_land", "handling_antigrav", "handling_water", "handling_glide",
    "traction", "miniturbo",
]


def _parse_equivalence_block(ws, start_row, end_row):
    """Parse a block of 'A = B = C' lines into {name: canonical_name}."""
    mapping = {}
    for r in range(start_row, end_row + 1):
        cell = ws.cell(row=r, column=1).value
        if not cell or not isinstance(cell, str) or "=" not in cell:
            continue
        names = [n.strip() for n in cell.split("=")]
        canonical = names[0]
        for n in names:
            mapping[n] = canonical
    return mapping


def load_stat_chart():
    """Parse reference/mk8_stat_chart_luigi_fan2.xlsx into stat lookup dicts.

    Returns dict of {part_name: {category: {stat_name: value}}} already
    expanded through the chart's equivalence groups, or None if the file
    isn't present (it's gitignored and not redistributed with the repo).
    """
    if not STAT_CHART_XLSX.exists():
        return None

    import openpyxl

    wb = openpyxl.load_workbook(STAT_CHART_XLSX, data_only=True)
    ws = wb["Sheet1"]

    def read_block(start_row, end_row):
        block = {}
        for r in range(start_row, end_row + 1):
            name = ws.cell(row=r, column=1).value
            if name is None:
                continue
            vals = [ws.cell(row=r, column=c).value for c in range(2, 14)]
            vals = [float(v) for v in vals]
            block[name] = dict(zip(STAT_COLUMNS, vals))
        return block

    characters = read_block(3, 11)
    bodies = read_block(13, 21)
    tires = read_block(23, 29)
    gliders = read_block(31, 32)

    char_equiv = _parse_equivalence_block(ws, 44, 52)
    body_equiv = _parse_equivalence_block(ws, 55, 62)
    tire_equiv = _parse_equivalence_block(ws, 66, 72)
    glider_equiv = _parse_equivalence_block(ws, 75, 76)

    def expand(base, equiv):
        expanded = dict(base)
        for alias, canonical in equiv.items():
            if canonical in base:
                expanded[alias] = base[canonical]
        return expanded

    return {
        "character": expand(characters, char_equiv),
        "body": expand(bodies, body_equiv),
        "tires": expand(tires, tire_equiv),
        "glider": expand(gliders, glider_equiv),
    }


def recompute_cbc(df, chart):
    """Recompute CBC_Speed/Accel/Handling/MiniTurbo/Traction from parts.

    In-game combined stat = character base + body delta + tire delta +
    glider delta. We use the "land" sub-stat for Speed and Handling since
    that's what the in-game build-screen stat bars display and what the
    original CBC_* columns describe (the terrain-specific sub-stats are
    handled separately as CBC_*_<Terrain> -- dropped elsewhere as leakage).
    """
    df = df.copy()
    recomputed = {k: [] for k in ["CBC_Speed", "CBC_Acceleration", "CBC_Handling", "CBC_Mini Turbo", "CBC_Traction"]}
    unverified = []
    diffs = []

    for _, row in df.iterrows():
        parts = [p.strip() for p in row["Community Best Combo"].split("+")]
        character, body, tires, glider = parts

        if chart is None or any(p in UNCOVERED_PARTS for p in parts):
            unverified.append(1)
            for k in recomputed:
                recomputed[k].append(row[k])
            continue

        # Color-swapped Yoshi (e.g. "Pink Yoshi", "Black Yoshi") has identical
        # stats to the base "Yoshi" entry -- the chart only lists unique
        # stat-lines, not every cosmetic recolor.
        if character != "Yoshi" and character.endswith("Yoshi"):
            character = "Yoshi"
        # Chart spells it "Mr Scooty" (no period); dataset uses "Mr. Scooty".
        if body == "Mr. Scooty":
            body = "Mr Scooty"

        try:
            c = chart["character"][character]
            b = chart["body"][body]
            t = chart["tires"][tires]
            g = chart["glider"][glider]
        except KeyError as e:
            # Part not found in chart/equivalence groups even though not in
            # the known UNCOVERED_PARTS list -- flag rather than guess.
            unverified.append(1)
            for k in recomputed:
                recomputed[k].append(row[k])
            continue

        speed = c["speed_land"] + b["speed_land"] + t["speed_land"] + g["speed_land"]
        handling = c["handling_land"] + b["handling_land"] + t["handling_land"] + g["handling_land"]
        accel = c["accel"] + b["accel"] + t["accel"] + g["accel"]
        traction = c["traction"] + b["traction"] + t["traction"] + g["traction"]
        miniturbo = c["miniturbo"] + b["miniturbo"] + t["miniturbo"] + g["miniturbo"]

        new_vals = {
            "CBC_Speed": speed, "CBC_Acceleration": accel, "CBC_Handling": handling,
            "CBC_Mini Turbo": miniturbo, "CBC_Traction": traction,
        }
        diffs.append({
            "Track Name": row["Track Name"], "Build": row["Community Best Combo"],
            **{f"{k}_old": row[k] for k in new_vals},
            **{f"{k}_new": v for k, v in new_vals.items()},
        })
        unverified.append(0)
        for k in recomputed:
            recomputed[k].append(new_vals[k])

    for k, vals in recomputed.items():
        df[k] = vals
    df["flag_target_unverified"] = unverified

    return df, pd.DataFrame(diffs)


# ---------------------------------------------------------------------------
# 2. Features: terrain, elevation, terrain shares, average speed.
# ---------------------------------------------------------------------------

def clean_terrain_and_elevation(df):
    """Fix Terrain Type, Elevation Changes, and terrain-share columns.

    Terrain Type: raw CSV has numeric codes (1/2/3) for rows 1-31 and text
    ("Road"/"Anti-Gravity") for rows 32-52. These are two disjoint sets of
    tracks -- no row appears in both encodings -- so there is no way to
    verify which code maps to which unobserved text label (batch B never
    uses "Water" or "Off-Road" as a label, so codes 2 and 3 are
    unverifiable). Rather than guess, we drop this categorical column
    entirely: the Off-Road/Road/Water/Anti-Grav percentage columns already
    carry the same information continuously and don't have this ambiguity.

    Elevation Changes: raw CSV has 0/5/10 for rows 1-31 and
    Low/Moderate/High for rows 32-52. Unlike Terrain Type this is ordinal
    with exactly 3 levels on each side, so 0->Low, 5->Moderate, 10->High is
    the only order-preserving mapping possible -- not a guess about which
    label means what, just the one consistent ordering. Encoded as
    elevation_level in {0, 1, 2}.

    Terrain shares (Off-Road/Road/Water/Anti-Grav): batch A are plain
    numbers, batch B are "NN%" strings. Both are percentages of the same
    0-100 scale (confirmed: rows sum to ~100 in 50/52 rows), so this is a
    parsing fix, not a rescaling. The two rows that don't sum to 100
    (N64 Rainbow Road: 105, Big Blue: 200 -- Big Blue is listed as both
    100% Water and 100% Anti-Grav, clearly a data entry error since the
    real track is 100% anti-gravity) are flagged rather than silently
    corrected.
    """
    df = df.copy()

    elevation_map_numeric = {"0": "Low", "5": "Moderate", "10": "High"}
    df["elevation_text"] = df["Elevation Changes"].astype(str).map(
        lambda v: elevation_map_numeric.get(v, v)
    )
    elevation_order = {"Low": 0, "Moderate": 1, "High": 2}
    df["elevation_level"] = df["elevation_text"].map(elevation_order)

    def pct(v):
        return float(str(v).replace("%", ""))

    for col, new_col in [
        ("Off-Road ()", "offroad_pct"), ("Road ()", "road_pct"),
        ("Water ()", "water_pct"), ("Anti-Grav ()", "antigrav_pct"),
    ]:
        df[new_col] = df[col].apply(pct)

    share_sum = df[["offroad_pct", "road_pct", "water_pct", "antigrav_pct"]].sum(axis=1)
    df["flag_terrain_pct_invalid"] = (share_sum != 100).astype(int)

    # Terrain Type dropped -- see docstring. Not kept even as a flagged
    # column since the percentage features supersede it.
    df = df.drop(columns=["Terrain Type", "Elevation Changes", "elevation_text",
                           "Off-Road ()", "Road ()", "Water ()", "Anti-Grav ()"])
    return df


def drop_average_speed(df):
    """Drop Average Speed (m/s) -- cannot be reconciled across batches.

    Batch A mean ~17.8 m/s, batch B mean ~9.4 m/s. This looked like it might
    be a simple 2x unit conversion, but checking implied race duration
    (Track Length * Lap Count / Average Speed) shows batch A times
    (~220-365s) and batch B times (~79-534s) are already roughly
    comparable in magnitude once track length is accounted for -- i.e. the
    raw speed difference is explained by batch B's genuinely shorter
    tracks, not a fixed unit-scale bug. Since there's no single verifiable
    conversion factor, per the "don't invent conversions" rule this column
    is dropped. See README for the with/without comparison this motivates.
    """
    return df.drop(columns=["Average Speed (m/s)"])


LEAKY_COLUMNS = [c for c in [
    "CBC_Speed_OffRoad", "CBC_Acceleration_OffRoad", "CBC_Handling_OffRoad",
    "CBC_Traction_OffRoad", "CBC_MiniTurbo_OffRoad",
    "CBC_Speed_Road", "CBC_Acceleration_Road", "CBC_Handling_Road",
    "CBC_Traction_Road", "CBC_MiniTurbo_Road",
    "CBC_Speed_Water", "CBC_Acceleration_Water", "CBC_Handling_Water",
    "CBC_Traction_Water", "CBC_MiniTurbo_Water",
    "CBC_Speed_AntiGrav", "CBC_Acceleration_AntiGrav", "CBC_Handling_AntiGrav",
    "CBC_Traction_AntiGrav", "CBC_MiniTurbo_AntiGrav",
]]


def main():
    df = pd.read_csv(RAW_CSV)
    df.columns = [c.strip() for c in df.columns]
    # The raw CSV's degree sign in this header is corrupted (already replaced
    # by U+FFFD at the byte level, not just a display issue) -- unrecoverable,
    # so renamed to a clean ASCII column name rather than guessing the symbol.
    angle_col = next(c for c in df.columns if c.startswith("Avg. Turn Angle"))
    df = df.rename(columns={angle_col: "Avg Turn Angle (deg)"})

    chart = load_stat_chart()
    df, diffs = recompute_cbc(df, chart)

    df = clean_terrain_and_elevation(df)
    df = drop_average_speed(df)
    df = df.drop(columns=[c for c in LEAKY_COLUMNS if c in df.columns])

    n_total = len(df)
    n_unverified = int(df["flag_target_unverified"].sum())
    n_recomputed = n_total - n_unverified

    print(f"Stat chart found: {chart is not None}")
    print(f"Rows: {n_total}  recomputed: {n_recomputed}  flagged unverified: {n_unverified}")
    print(f"Rows with invalid terrain-share sum: {int(df['flag_terrain_pct_invalid'].sum())}")

    target_cols = ["CBC_Speed", "CBC_Acceleration", "CBC_Handling", "CBC_Mini Turbo", "CBC_Traction"]
    n_unique_targets = df[target_cols].drop_duplicates().shape[0]
    print(f"Unique target vectors across {n_total} tracks: {n_unique_targets}")

    if not diffs.empty:
        changed = diffs[
            (diffs["CBC_Speed_old"] != diffs["CBC_Speed_new"]) |
            (diffs["CBC_Acceleration_old"] != diffs["CBC_Acceleration_new"]) |
            (diffs["CBC_Handling_old"] != diffs["CBC_Handling_new"]) |
            (diffs["CBC_Mini Turbo_old"] != diffs["CBC_Mini Turbo_new"]) |
            (diffs["CBC_Traction_old"] != diffs["CBC_Traction_new"])
        ]
        print(f"\nRows where recomputed CBC values differ from raw CSV: {len(changed)} / {len(diffs)} recomputed rows")
        diffs.to_csv(ROOT / "data" / "mk8_cbc_recompute_diff.csv", index=False)
        print(f"Full diff written to data/mk8_cbc_recompute_diff.csv")

    df.to_csv(CLEAN_CSV, index=False)
    print(f"\nWrote {CLEAN_CSV}")


if __name__ == "__main__":
    main()
