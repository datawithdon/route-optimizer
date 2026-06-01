# Route Optimizer

Production-grade **Capacitated Vehicle Routing Problem (CVRP)** solver. Geocodes a list of delivery addresses, builds a distance matrix, optimises multi-vehicle routes, and renders an interactive HTML map.

---

## How it works

| Step | What happens |
|---|---|
| **1 · Geocode** | Addresses resolved to lat/lon via Nominatim (OpenStreetMap) |
| **2 · Distance matrix** | Haversine great-circle distances (or OSRM road-network, toggle-on) |
| **3 · Solve CVRP** | OR-Tools Guided Local Search — near-optimal in seconds |
| **4 · Fallback** | Nearest-neighbour + 2-opt if OR-Tools is not installed |
| **5 · Visualise** | Animated per-vehicle routes on an interactive Folium map |
| **6 · Export** | CSV with `stop_sequence`, `vehicle_id`, `arrival_distance_milestone_km` |

---

## Quick start

```bash
pip install -r requirements.txt
python optimizer.py
```

Outputs land in `vrp_output/`:

| File | Description |
|---|---|
| `optimized_routes.html` | Interactive map — open in any browser |
| `optimized_stops.csv` | Full stop sequence table |

---

## Configuration

Edit the top of `optimizer.py`:

```python
# Where routes originate
DEPOT = {
    "name":    "Central Warehouse",
    "address": "233 S Wacker Dr, Chicago, IL 60606",
    "lat":     41.87893,
    "lon":     -87.63590,
}

NUM_VEHICLES     = 4      # available drivers
VEHICLE_CAPACITY = 150    # max load per vehicle (units / kg)
MAX_ROUTE_KM     = 80.0   # per-route distance cap (0 = uncapped)
SOLVER_TIME_SEC  = 30     # OR-Tools time budget
```

Replace `RAW_STOPS` with your own destinations:

```python
RAW_STOPS = [
    {"name": "Stop A", "address": "123 Main St, Chicago, IL", "demand": 20},
    {"name": "Stop B", "address": "456 Oak Ave, Evanston, IL", "demand": 35},
    # ...
]
```

### Road-network distances (optional)

```python
USE_ROAD_NETWORK = True   # calls OSRM instead of Haversine
OSRM_BASE_URL    = "https://your-osrm-server.com"
OSRM_ALLOWED_HOSTS = frozenset({"your-osrm-server.com"})
```

The public OSRM demo server works for small tests. Self-host for production.

---

## Console output

```
──────────────────────────────────────────────────────────
  ROUTE OPTIMISATION SUMMARY
──────────────────────────────────────────────────────────
  Solver            :  OR-Tools GLS (CVRP)
  Status            :  SUCCESS
  Wall-clock time   :  4.21 s
──────────────────────────────────────────────────────────
  Stops geocoded    :  18 / 18
  Vehicles used     :  4 / 4
  Total demand      :  412 units
  Vehicle capacity  :  150 units each
  Avg cap. util.    :  68.7 %
──────────────────────────────────────────────────────────
  Total distance    :  187.43 km
  Avg / route       :  46.86 km
──────────────────────────────────────────────────────────
  V1:   5 stops  |   41.20 km  |  load 103/150  (69%)
  V2:   5 stops  |   52.18 km  |  load 107/150  (71%)
  V3:   4 stops  |   48.91 km  |  load  96/150  (64%)
  V4:   4 stops  |   45.14 km  |  load 106/150  (71%)
──────────────────────────────────────────────────────────
```

---

## Map preview

Each vehicle gets a distinct colour. Animated arrows show direction of travel. Numbered badges mark stop sequence within each route. A floating legend summarises distance and stop counts.

---

## Dependencies

| Package | Purpose |
|---|---|
| `ortools` | CVRP solver (Guided Local Search) |
| `folium` | Interactive HTML map |
| `geopy` | Nominatim geocoding |
| `pandas` / `numpy` | Data handling & matrix math |
| `requests` | OSRM API calls (optional) |
| `geopandas` / `shapely` | Geospatial projections |

---

## Security

Three audit rounds were applied before the initial commit. Mitigations include:

- **SSRF** — OSRM URL validated against an explicit `frozenset` allowlist; `https://` enforced; embedded credentials rejected
- **HTML injection** — `html.escape()` on all user-controlled tooltip strings
- **CSV formula injection** (CWE-1236) — first-character sanitiser on stop names and addresses
- **Path traversal + TOCTOU** — output paths resolved and checked against `cwd()`; symlink guard called immediately before each file write
- **Log injection** (CWE-117) — CR/LF/TAB neutralised in every log call site; null bytes and C0 control chars rejected at input validation
- **OSRM response validation** — shape, NaN/Inf, sign, and 2,000-node size cap enforced before numpy conversion
- **NaN propagation** — `np.isfinite` check before OR-Tools int64 scaling (NaN silently passes `> INT64_MAX`)
- **Haversine domain error** — accumulator clamped to `[0, 1]` to prevent `sqrt(negative)` on near-antipodal pairs
- **Config guards** — depot coordinates, `NUM_VEHICLES`, `VEHICLE_CAPACITY`, `MAX_ROUTE_KM`, `SOLVER_TIME_SEC` all validated before any I/O; `bool` subclass bypass blocked
- **DoS** — 2-opt iteration cap, matrix node cap, `MAX_ROUTE_KM` upper bound

---

## License

MIT
