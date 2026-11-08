# Data Notice

## Dataset ownership

`data/mk8_tracks.csv` and its cleaned derivative `data/mk8_tracks_clean.csv` are
my own compilation, built by hand by measuring and recording track properties
(length, turns, elevation change, terrain composition) from gameplay, and
by recording each track's community-agreed best build.

## Mario Kart 8 game data

Mario Kart 8 and Mario Kart 8 Deluxe, all character names, kart part names,
track names, and their associated in-game statistics are the property of
Nintendo. This repository uses them for non-commercial, educational
analysis only. No game assets (models, textures, audio) are included or
redistributed.

## Stat chart credit

Build-stat recomputation (`src/clean.py`) uses part stat values and
equivalence groupings from **Luigi_Fan2's Mario Kart 8 stat chart**, a
community reference compiling in-game character/kart/tire/glider statistics
for the original Wii U release of Mario Kart 8.

The stat chart file itself is **not included or redistributed in this
repository** (see `.gitignore`) -- only the numeric results of running it
through `src/clean.py` are committed, as derived data. If you want to
reproduce the recomputation step yourself, you will need to source your own
copy of the chart and place it at `reference/mk8_stat_chart_luigi_fan2.xlsx`.

## Known limitation: Wii U vs. Deluxe stats

The stat chart covers the original Wii U Mario Kart 8 roster and parts.
Mario Kart 8 Deluxe later patched some stat values and added new characters
and parts. Seven rows in this dataset use Deluxe-exclusive characters or
kart parts (Birdo, Funky Kong, Kamek, Peachette, Inkstriker) not covered by
the chart; these rows are flagged (`flag_target_unverified=1`) and retain
their original, unverified values rather than a guessed recomputation. See
the README's Limitations and Next Steps sections for detail.
