"""Legacy on-disk keys used `car_*`; code uses `robot_*`. Bridge when reading files."""


def canonical_stat_dict(raw: dict) -> dict:
    """Copy stats so every `robot_*` key exists: reuse value from `car_*` when needed."""
    out = dict(raw)
    for key, value in list(raw.items()):
        if not key.startswith("car_"):
            continue
        robot_key = "robot_" + key[4:]  # strip leading "car_"
        if robot_key not in out:
            out[robot_key] = value
    return out


def get_solver_x_value(x: dict, logical_key: str):
    """Read a channel from solver iterate dict `x`; prefer `robot_*`, fall back to `car_*`."""
    if logical_key in x:
        return x[logical_key]
    if logical_key.startswith("robot_"):
        legacy_key = "car_" + logical_key[len("robot_") :]
        if legacy_key in x:
            return x[legacy_key]
    raise KeyError(logical_key)
