# Changelog

## v1.1.0 (2026-11-28)

### Added
- `curve_fit.py`: AE keyframe export. With `--out`, every fit also writes `prefix_keys.json` (2 keys with times,
  values, influence and speed) and `prefix_keys.jsx`, which puts those keys on the selected layers at the playhead.
  `fit --all` fits every property that moved and writes one graph and one keys file per property.
  `series --prop scale|rot|x|y` says which transform property the values belong to.
- `ramp_profile.py`: per-frame speed-ramp profiles (apex, beat, step) with CSV/JSON export. Ramps are clamped so they
  never read past the end of the source.
- `beat_map.py`: `--grid beat|8th|16th` and `--csv` marker export (HH:MM:SS:FF at `--fps`) for Premiere.
- pytest smoke tests for every Python tool, all on synthetic input (no media in the repo).

### Fixed
- `beat_map.py`: one transient counted as several hits.
- `curve_fit.py`: `--prop auto` measured the range of NaN-padded tracks (bbox mode) as if the gaps were zeros.
  An unknown `--prop` or command now exits with an error.

## v1.0.0 (2026-11-12)
First release: `datamosh.py`, `curve_fit.py`, `grade_match.py`, `beat_map.py`, `keyalpha.py`, `voice_fx.py`, Swift tools.
