#!/usr/bin/env python3
"""
Production-Grade Capacitated Vehicle Routing Problem (CVRP) Optimizer
======================================================================

Optimization Approach
---------------------
PRIMARY  — Google OR-Tools: Guided Local Search (GLS) metaheuristic on a CVRP.
           GLS escapes local optima by penalising frequently-used arc features.
           Seeded with PATH_CHEAPEST_ARC for a fast first feasible solution.
           Achieves near-optimal results on sub-1000-node instances within seconds.

FALLBACK — Nearest-Neighbor + 2-opt: Greedy construction with local improvement.
           Activates automatically if OR-Tools is not installed. No extra deps.

Distance Model
--------------
Haversine great-circle (EPSG:4326) by default.
Set USE_ROAD_NETWORK = True to switch to OSRM road-network API calls.

Dependencies
------------
    pip install pandas numpy folium geopy ortools scipy requests
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

import html as html_mod
import math
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import folium
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderServiceError, GeocoderTimedOut, GeocoderUnavailable
from geopy.geocoders import Nominatim

try:
    from folium.plugins import AntPath
    _ANTPATH_AVAILABLE = True
except ImportError:
    _ANTPATH_AVAILABLE = False

try:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False
    warnings.warn(
        "OR-Tools not installed (`pip install ortools`). "
        "Falling back to nearest-neighbour + 2-opt heuristic.",
        stacklevel=2,
    )

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-7s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — CONFIGURATION  (edit this section to match your problem)
# ─────────────────────────────────────────────────────────────────────────────

# ── Depot ─────────────────────────────────────────────────────────────────────
DEPOT: Dict = {
    "name":    "Central Distribution Warehouse",
    "address": "233 S Wacker Dr, Chicago, IL 60606",   # Willis Tower area
    "lat":     41.87893,
    "lon":     -87.63590,
}

# ── Fleet ─────────────────────────────────────────────────────────────────────
NUM_VEHICLES:     int   = 4      # available drivers / vehicles
VEHICLE_CAPACITY: int   = 150    # max load per vehicle (units, kg, etc.)
MAX_ROUTE_KM:     float = 80.0   # max one-way route distance per vehicle (0 = uncapped)

# ── Solver ────────────────────────────────────────────────────────────────────
SOLVER_TIME_SEC:  int  = 30      # OR-Tools wall-clock time budget
USE_ROAD_NETWORK: bool = False   # True → OSRM API; False → Haversine
OSRM_BASE_URL:    str  = "https://router.project-osrm.org"

# Allowlist of trusted OSRM hostnames. Add your self-hosted domain here.
# frozenset prevents accidental mutation (e.g. .append(None)) that could
# let a None-hostname URL slip through the membership check.
OSRM_ALLOWED_HOSTS: frozenset = frozenset({"router.project-osrm.org"})

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_DIR:   Path = Path("vrp_output")
MAP_HTML:     Path = OUTPUT_DIR / "optimized_routes.html"
STOPS_CSV:    Path = OUTPUT_DIR / "optimized_stops.csv"

# ── Per-vehicle colour palette ────────────────────────────────────────────────
ROUTE_COLORS: List[str] = [
    "#E74C3C",   # vehicle 1 — red
    "#2980B9",   # vehicle 2 — blue
    "#27AE60",   # vehicle 3 — green
    "#F39C12",   # vehicle 4 — amber
    "#8E44AD",   # vehicle 5 — purple
    "#16A085",   # vehicle 6 — teal
    "#D35400",   # vehicle 7 — burnt orange
    "#1ABC9C",   # vehicle 8 — emerald
]

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — INPUT DATASET
# ─────────────────────────────────────────────────────────────────────────────
# Replace or extend this list with your real destinations.
# Each entry: name (str), address (str), demand (int — units / kg per stop).

RAW_STOPS: List[Dict] = [
    {"name": "Millennium Park",        "address": "201 E Randolph St, Chicago, IL 60602",       "demand": 20},
    {"name": "Navy Pier",              "address": "600 E Grand Ave, Chicago, IL 60611",          "demand": 35},
    {"name": "Wrigley Field",          "address": "1060 W Addison St, Chicago, IL 60613",        "demand": 15},
    {"name": "Lincoln Park Zoo",       "address": "2001 N Clark St, Chicago, IL 60614",          "demand": 28},
    {"name": "Museum of Science",      "address": "5700 S Lake Shore Dr, Chicago, IL 60637",     "demand": 40},
    {"name": "Hyde Park Post Office",  "address": "1526 E 55th St, Chicago, IL 60615",           "demand": 22},
    {"name": "Midway Airport",         "address": "5700 S Cicero Ave, Chicago, IL 60638",        "demand": 18},
    {"name": "Oak Park Center",        "address": "1 Harlem Ave, Oak Park, IL 60301",            "demand": 30},
    {"name": "Evanston North Shore",   "address": "1603 Orrington Ave, Evanston, IL 60201",      "demand": 25},
    {"name": "Rosemont Convention",    "address": "5555 N River Rd, Rosemont, IL 60018",         "demand": 12},
    {"name": "Schaumburg Woodfield",   "address": "5 Woodfield Mall, Schaumburg, IL 60173",      "demand": 45},
    {"name": "Naperville Hub",         "address": "200 E 5th Ave, Naperville, IL 60563",         "demand": 33},
    {"name": "Joliet Depot",           "address": "50 W Jefferson St, Joliet, IL 60432",         "demand": 27},
    {"name": "Orland Park Mall",       "address": "288 Orland Square Dr, Orland Park, IL 60462", "demand": 19},
    {"name": "Pilsen Market",          "address": "1800 S Halsted St, Chicago, IL 60608",        "demand": 14},
    {"name": "Wicker Park Depot",      "address": "1375 N Milwaukee Ave, Chicago, IL 60622",     "demand": 21},
    {"name": "Bridgeport Warehouse",   "address": "3300 S Halsted St, Chicago, IL 60608",        "demand": 17},
    {"name": "Calumet City Shop",      "address": "1919 Sibley Blvd, Calumet City, IL 60409",    "demand": 11},
]

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — INPUT VALIDATION & GEOCODING
# ─────────────────────────────────────────────────────────────────────────────

_MAX_DEMAND        = 1_000_000   # sanity cap — adjust to your unit scale
_MAX_ADDRESS_LEN   = 500
_MAX_NAME_LEN      = 200
_LOG_CONTROL_CHARS = str.maketrans({"\n": "\\n", "\r": "\\r", "\t": "\\t"})

# C0/C1 control characters (excluding the printable \t \n \r that _safe_log
# neutralises) that must be rejected in stop names/addresses. Null bytes
# truncate Nominatim queries; others corrupt CSV files and HTML rendering.
_FORBIDDEN_FIELD_CHARS: frozenset = frozenset(
    chr(c) for c in range(0x20)        # U+0000–U+001F (C0 controls)
    if chr(c) not in ("\t", "\n", "\r")  # \t/\n/\r handled by _safe_log
) | {"\x7f"}                             # DEL


def _safe_log(text: str) -> str:
    """Neutralise CR/LF/TAB to prevent log-line forging (CWE-117)."""
    return text.translate(_LOG_CONTROL_CHARS)


def validate_raw_stops(stops: List[Dict]) -> None:
    """
    Validate the structure and values of RAW_STOPS before processing.
    Raises TypeError or ValueError on the first invalid entry found.
    This prevents malformed input from propagating into the solver or
    being silently written to output files.
    """
    if not stops:
        raise ValueError("RAW_STOPS is empty — provide at least one delivery stop.")

    required_keys = {"name", "address", "demand"}
    for i, stop in enumerate(stops):
        if not isinstance(stop, dict):
            raise TypeError(f"RAW_STOPS[{i}]: expected dict, got {type(stop).__name__}")

        missing = required_keys - stop.keys()
        if missing:
            raise KeyError(f"RAW_STOPS[{i}]: missing required keys {sorted(missing)}")

        name, address, demand = stop["name"], stop["address"], stop["demand"]

        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"RAW_STOPS[{i}]: 'name' must be a non-empty string")
        if len(name) > _MAX_NAME_LEN:
            raise ValueError(f"RAW_STOPS[{i}]: 'name' exceeds {_MAX_NAME_LEN} characters")
        bad = _FORBIDDEN_FIELD_CHARS & set(name)
        if bad:
            raise ValueError(
                f"RAW_STOPS[{i}]: 'name' contains forbidden control characters "
                f"{sorted(repr(c) for c in bad)}"
            )

        if not isinstance(address, str) or not address.strip():
            raise ValueError(f"RAW_STOPS[{i}]: 'address' must be a non-empty string")
        if len(address) > _MAX_ADDRESS_LEN:
            raise ValueError(f"RAW_STOPS[{i}]: 'address' exceeds {_MAX_ADDRESS_LEN} characters")
        bad = _FORBIDDEN_FIELD_CHARS & set(address)
        if bad:
            raise ValueError(
                f"RAW_STOPS[{i}]: 'address' contains forbidden control characters "
                f"{sorted(repr(c) for c in bad)}"
            )

        if not isinstance(demand, int) or isinstance(demand, bool):
            raise TypeError(f"RAW_STOPS[{i}]: 'demand' must be an integer, got {type(demand).__name__}")
        if demand < 0:
            raise ValueError(f"RAW_STOPS[{i}]: 'demand' must be >= 0 (got {demand})")
        if demand > _MAX_DEMAND:
            raise ValueError(f"RAW_STOPS[{i}]: 'demand' {demand} exceeds maximum {_MAX_DEMAND}")


class GeoPoint:
    """Holds a resolved delivery stop with coordinates and demand."""

    __slots__ = ("name", "address", "demand", "lat", "lon", "geocoded", "error")

    def __init__(self, name: str, address: str, demand: int) -> None:
        self.name:     str            = name
        self.address:  str            = address
        self.demand:   int            = demand
        self.lat:      Optional[float] = None
        self.lon:      Optional[float] = None
        self.geocoded: bool            = False
        self.error:    Optional[str]   = None


def geocode_stops(raw_stops: List[Dict]) -> List[GeoPoint]:
    """
    Geocode a list of stop dicts → GeoPoint objects via Nominatim (OSM).

    Rate-limited to 1.2 req/s to comply with the Nominatim usage policy.
    Failed stops are logged as warnings and excluded from routing — they
    do NOT raise an exception so the rest of the run proceeds normally.
    """
    geolocator = Nominatim(user_agent="vrp_optimizer_prod_v2", timeout=12)
    geocode = RateLimiter(
        geolocator.geocode,
        min_delay_seconds=1.2,
        error_wait_seconds=6.0,
        max_retries=3,
        return_value_on_exception=None,
    )
    points: List[GeoPoint] = []

    for raw in raw_stops:
        pt = GeoPoint(name=raw["name"], address=raw["address"], demand=raw["demand"])
        try:
            result = geocode(raw["address"], addressdetails=False, language="en")
            if result is not None:
                rlat, rlon = result.latitude, result.longitude
                if not (math.isfinite(rlat) and math.isfinite(rlon)):
                    pt.error = "geocoder returned non-finite coordinates"
                    log.warning("  --  %-30s  non-finite coords from geocoder", _safe_log(pt.name))
                elif not (-90.0 <= rlat <= 90.0 and -180.0 <= rlon <= 180.0):
                    pt.error = f"geocoder returned out-of-range coordinates ({rlat}, {rlon})"
                    log.warning("  --  %-30s  out-of-range coords from geocoder", _safe_log(pt.name))
                else:
                    pt.lat      = rlat
                    pt.lon      = rlon
                    pt.geocoded = True
                    log.info("  OK  %-30s  (%.5f, %.5f)", _safe_log(pt.name), pt.lat, pt.lon)
            else:
                pt.error = "geocoder returned None (address not found)"
                log.warning("  --  %-30s  geocoder returned no result", _safe_log(pt.name))
        except (GeocoderTimedOut, GeocoderServiceError, GeocoderUnavailable) as exc:
            # Store only the exception class name, not the full message.
            # Full geopy exception strings include the query URL which
            # contains the percent-encoded address (PII).
            pt.error = type(exc).__name__
            log.error("  ERR %-30s  geocoding failed (%s)", _safe_log(pt.name), pt.error)

        points.append(pt)

    n_failed = sum(1 for p in points if not p.geocoded)
    if n_failed:
        log.warning(
            "%d/%d stop(s) failed geocoding and are excluded: %s",
            n_failed, len(points),
            [_safe_log(p.name) for p in points if not p.geocoded],
        )

    return points


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — DISTANCE / COST MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine great-circle distance in kilometres (EPSG:4326)."""
    R = 6_371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi    = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (math.sin(d_phi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2)
    # Clamp to [0, 1]: IEEE-754 rounding can push `a` just above 1.0 for
    # near-antipodal pairs, making sqrt(1 - a) raise ValueError.
    a = max(0.0, min(1.0, a))
    return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


_MAX_MATRIX_NODES = 2_000   # matches OSRM cap; 2000² matrix ≈ 32 MB


def _build_haversine_matrix(coords: List[Tuple[float, float]]) -> np.ndarray:
    """Return symmetric N×N distance matrix (km) from (lat, lon) list."""
    n = len(coords)
    if n > _MAX_MATRIX_NODES:
        raise ValueError(
            f"Stop count {n} exceeds the matrix node cap of {_MAX_MATRIX_NODES}. "
            "Reduce RAW_STOPS or increase _MAX_MATRIX_NODES / switch to OSRM."
        )
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d        = haversine_km(*coords[i], *coords[j])
            mat[i][j] = d
            mat[j][i] = d
    return mat


def _validate_osrm_url(base_url: str) -> str:
    """
    Guard against SSRF by enforcing scheme=https and an explicit hostname
    allowlist. Raises ValueError for any URL that doesn't pass both checks.
    Add trusted self-hosted OSRM domains to OSRM_ALLOWED_HOSTS.
    """
    parsed = urlparse(base_url)
    if parsed.scheme != "https":
        raise ValueError(
            f"OSRM URL must use HTTPS (got scheme='{parsed.scheme}'). "
            "Update OSRM_BASE_URL to start with 'https://'."
        )
    # Block embedded credentials (e.g. https://user:pass@host) — credential leakage risk
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(
            "OSRM_BASE_URL must not contain embedded credentials (user:password@host). "
            "Pass auth via request headers or environment variables instead."
        )
    # Explicit None guard: urlparse("https://") returns hostname=None;
    # without this check a None entry in the allowlist would let it pass.
    if parsed.hostname is None or parsed.hostname not in OSRM_ALLOWED_HOSTS:
        raise ValueError(
            f"OSRM hostname '{parsed.hostname}' is not in OSRM_ALLOWED_HOSTS. "
            "Add it explicitly if this is your trusted self-hosted server."
        )
    return base_url


def _build_osrm_matrix(
    coords: List[Tuple[float, float]], base_url: str
) -> np.ndarray:
    """
    Road-network distance matrix via OSRM /table/v1/driving endpoint.
    Returns distances in km. Falls back to Haversine on connection/API errors.
    Note: the public OSRM demo server caps payloads; self-host OSRM for
    production workloads with many stops.
    """
    try:
        import requests  # optional dep
        from requests.exceptions import RequestException

        _validate_osrm_url(base_url)   # SSRF guard — raises on untrusted URL

        coord_str = ";".join(f"{lon},{lat}" for lat, lon in coords)
        url       = f"{base_url}/table/v1/driving/{coord_str}?annotations=distance"
        resp      = requests.get(url, timeout=30)
        resp.raise_for_status()
        payload   = resp.json()

        if payload.get("code") == "Ok":
            try:
                raw = payload["distances"]
                if not isinstance(raw, list):
                    raise TypeError(f"distances must be a list, got {type(raw).__name__}")

                # Memory-exhaustion guard: reject unreasonably large payloads
                _MAX_NODES = 2_000
                if len(raw) > _MAX_NODES:
                    raise ValueError(
                        f"OSRM returned {len(raw)} rows; refusing matrices larger than {_MAX_NODES}."
                    )

                mat = np.array(raw, dtype=np.float64)

                if mat.ndim != 2:
                    raise ValueError(f"distances must be 2-D, got shape {mat.shape}")
                expected = (len(coords), len(coords))
                if mat.shape != expected:
                    raise ValueError(f"distances shape {mat.shape} != expected {expected}")
                if not np.all(np.isfinite(mat)):
                    raise ValueError("distances matrix contains NaN or Inf values")
                if np.any(mat < 0):
                    raise ValueError("distances matrix contains negative values")

            except (TypeError, ValueError, KeyError) as exc:
                log.warning("OSRM response validation failed (%s: %s). Falling back to Haversine.",
                            type(exc).__name__, exc)
                return _build_haversine_matrix(coords)

            mat_km = mat / 1_000.0   # m → km
            log.info("OSRM distance matrix: %d × %d nodes", len(coords), len(coords))
            return mat_km

        # Log only the API status code — never the raw server message (info disclosure)
        log.warning("OSRM API returned non-OK status: %s. Falling back to Haversine.",
                    payload.get("code", "UNKNOWN"))
        return _build_haversine_matrix(coords)

    except ValueError:
        # Re-raise configuration errors (bad URL, untrusted host) — do not silently fall back
        raise
    except (RequestException, KeyError, TypeError) as exc:
        log.warning("OSRM request failed (type: %s). Falling back to Haversine.",
                    type(exc).__name__)
        return _build_haversine_matrix(coords)


def build_distance_matrix(
    coords: List[Tuple[float, float]],
    use_road_network: bool = False,
    osrm_url: str = "",
) -> np.ndarray:
    """Public entry point: dispatch to OSRM or Haversine."""
    if use_road_network:
        return _build_osrm_matrix(coords, osrm_url)
    return _build_haversine_matrix(coords)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — OR-TOOLS CVRP SOLVER
# ─────────────────────────────────────────────────────────────────────────────

def solve_cvrp_ortools(
    dist_matrix:      np.ndarray,
    demands:          List[int],
    num_vehicles:     int,
    vehicle_capacity: int,
    max_route_km:     float,
    time_limit_sec:   int,
) -> Optional[Dict]:
    """
    Solve CVRP with OR-Tools Guided Local Search.

    Returns
    -------
    dict with keys:
        routes          — list[list[int]]  (0 = depot index)
        route_distances — list[float]  km per vehicle
        total_distance  — float  total km across all vehicles
        status          — str  solver status label
    Returns None if OR-Tools is unavailable or no feasible solution found.
    """
    if not ORTOOLS_AVAILABLE:
        return None

    # OR-Tools requires integer costs; scale distances to preserve two decimal places.
    # Use int64 throughout and check that scaled values fit within signed 64-bit range
    # to prevent silent overflow from yielding incorrect route costs.
    SCALE = 100
    # NaN check MUST come before max() — NaN > INT64_MAX evaluates False,
    # silently passing the guard and producing garbage int64 values (platform-
    # dependent; numpy casts NaN→0 on most systems, corrupting all arc costs).
    if not np.all(np.isfinite(dist_matrix)):
        raise ValueError(
            "Distance matrix contains NaN or Inf values. "
            "Verify that all coordinates are valid and geocoding succeeded."
        )
    scaled_max = dist_matrix.max() * SCALE
    INT64_MAX  = 2**63 - 1
    if scaled_max > INT64_MAX:
        raise OverflowError(
            f"Scaled distance {scaled_max:.0f} exceeds int64 range. "
            "Check that coordinates are valid and distances are in km."
        )
    int_mat = (dist_matrix * SCALE).astype(np.int64).tolist()
    n_nodes   = len(demands)

    manager = pywrapcp.RoutingIndexManager(n_nodes, num_vehicles, 0)
    model   = pywrapcp.RoutingModel(manager)

    # ── Arc cost callback ──────────────────────────────────────────────────────
    def _dist_cb(from_idx: int, to_idx: int) -> int:
        return int_mat[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)]

    transit_idx = model.RegisterTransitCallback(_dist_cb)
    model.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    # ── Capacity dimension ─────────────────────────────────────────────────────
    def _demand_cb(from_idx: int) -> int:
        return demands[manager.IndexToNode(from_idx)]

    demand_idx = model.RegisterUnaryTransitCallback(_demand_cb)
    model.AddDimensionWithVehicleCapacity(
        demand_idx,
        0,                                      # no slack
        [vehicle_capacity] * num_vehicles,
        True,                                   # start cumul at 0
        "Capacity",
    )

    # ── Max-route-distance dimension (optional) ────────────────────────────────
    if max_route_km > 0:
        model.AddDimension(
            transit_idx,
            0,
            int(max_route_km * SCALE),
            True,
            "Distance",
        )
        dist_dim = model.GetDimensionOrDie("Distance")
        # Penalise span imbalance between vehicles (promotes load balancing)
        dist_dim.SetGlobalSpanCostCoefficient(100)

    # ── Search parameters ──────────────────────────────────────────────────────
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    params.time_limit.seconds = time_limit_sec
    params.log_search          = False   # set True to see solver iterations

    # ── Solve ──────────────────────────────────────────────────────────────────
    solution = model.SolveWithParameters(params)

    _STATUS = {
        0: "NOT_SOLVED", 1: "SUCCESS",
        2: "PARTIAL_SUCCESS_LOCAL_OPTIMUM_NOT_REACHED",
        3: "FAIL", 4: "FAIL_TIMEOUT", 5: "INVALID",
    }
    status_label = _STATUS.get(model.status(), "UNKNOWN")
    log.info("OR-Tools status: %s", status_label)

    if solution is None:
        log.error("No feasible solution found. Check capacity / max-distance constraints.")
        return None

    routes, dists = [], []
    for v in range(num_vehicles):
        route, idx = [], model.Start(v)
        while not model.IsEnd(idx):
            route.append(manager.IndexToNode(idx))
            idx = solution.Value(model.NextVar(idx))
        route.append(manager.IndexToNode(idx))   # close at depot

        if len(route) > 2:   # skip empty vehicle routes (depot → depot only)
            total_d = sum(
                dist_matrix[route[k]][route[k + 1]] for k in range(len(route) - 1)
            )
            routes.append(route)
            dists.append(round(total_d, 3))

    return {
        "routes":          routes,
        "route_distances": dists,
        "total_distance":  round(sum(dists), 3),
        "status":          status_label,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4b — FALLBACK: NEAREST-NEIGHBOUR + 2-OPT HEURISTIC
# ─────────────────────────────────────────────────────────────────────────────

_TWO_OPT_MAX_ITER = 500   # prevents CPU exhaustion on large routes (DoS guard)


def _two_opt_improve(route: List[int], dm: np.ndarray) -> List[int]:
    """
    Apply 2-opt local search to a single route.
    Capped at _TWO_OPT_MAX_ITER outer passes to bound worst-case CPU time.
    """
    best     = route[:]
    improved = True
    iters    = 0
    while improved and iters < _TWO_OPT_MAX_ITER:
        improved = False
        iters   += 1
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                # Cost delta for reversing the segment best[i..j]
                delta = (
                    (dm[best[i - 1]][best[j]]   + dm[best[i]][best[j + 1]])
                    - (dm[best[i - 1]][best[i]] + dm[best[j]][best[j + 1]])
                )
                if delta < -1e-10:
                    best[i : j + 1] = best[i : j + 1][::-1]
                    improved = True
    if iters >= _TWO_OPT_MAX_ITER:
        log.debug("2-opt reached iteration cap (%d) — solution may not be locally optimal.",
                  _TWO_OPT_MAX_ITER)
    return best


def solve_cvrp_nearest_neighbor(
    dist_matrix:      np.ndarray,
    demands:          List[int],
    num_vehicles:     int,
    vehicle_capacity: int,
    max_route_km:     float,
) -> Dict:
    """
    Greedy nearest-neighbour construction heuristic for CVRP,
    followed by intra-route 2-opt improvement.

    This is dependency-free and always terminates, but produces
    sub-optimal solutions compared to OR-Tools. Use as a fallback only.
    """
    unvisited: set = set(range(1, len(demands)))   # exclude depot (node 0)
    routes, dists  = [], []

    for _ in range(num_vehicles):
        if not unvisited:
            break

        route    = [0]
        load     = 0
        dist_acc = 0.0

        while unvisited:
            last = route[-1]
            # Candidates: demand fits AND (if capped) remaining distance allows return
            candidates = [
                j for j in unvisited
                if load + demands[j] <= vehicle_capacity
                and (
                    max_route_km <= 0
                    or dist_acc + dist_matrix[last][j] + dist_matrix[j][0] <= max_route_km
                )
            ]
            if not candidates:
                break
            nearest = min(candidates, key=lambda j: dist_matrix[last][j])
            route.append(nearest)
            dist_acc += dist_matrix[last][nearest]
            load     += demands[nearest]
            unvisited.discard(nearest)

        route.append(0)   # return to depot

        if len(route) > 2:
            route   = _two_opt_improve(route, dist_matrix)
            total_d = sum(
                dist_matrix[route[k]][route[k + 1]] for k in range(len(route) - 1)
            )
            routes.append(route)
            dists.append(round(total_d, 3))

    if unvisited:
        log.warning(
            "Heuristic: %d stop(s) could not be assigned (constraints too tight): %s",
            len(unvisited), sorted(unvisited),
        )

    return {
        "routes":          routes,
        "route_distances": dists,
        "total_distance":  round(sum(dists), 3),
        "status":          "HEURISTIC_2OPT_COMPLETE",
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — FOLIUM MAP VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def _node_map(active: List[GeoPoint]) -> Dict[int, GeoPoint]:
    """Return {node_index: GeoPoint} where node 0 is the depot."""
    return {i + 1: p for i, p in enumerate(active)}


def build_folium_map(
    depot:        Dict,
    active:       List[GeoPoint],
    solution:     Dict,
    colors:       List[str],
) -> folium.Map:
    """
    Render an interactive Folium map:
      - Depot       → red 'home' icon marker
      - Stops       → steelblue circle markers with tooltips
      - Routes      → animated AntPath polylines (or PolyLine fallback),
                      colour-coded per vehicle
      - Sequence    → small numbered badges overlaid on each stop
      - Legend      → floating HTML panel (bottom-left)
    """
    idx_to_pt = _node_map(active)

    all_lats = [depot["lat"]] + [p.lat for p in active]
    all_lons = [depot["lon"]] + [p.lon for p in active]
    center   = [float(np.mean(all_lats)), float(np.mean(all_lons))]

    fmap = folium.Map(
        location=center,
        zoom_start=10,
        tiles="CartoDB positron",
        control_scale=True,
    )

    # ── Depot marker ───────────────────────────────────────────────────────────
    folium.Marker(
        location=[depot["lat"], depot["lon"]],
        tooltip=folium.Tooltip(
            f"<b>DEPOT</b><br>{html_mod.escape(depot['name'])}"
            f"<br><i>{html_mod.escape(depot['address'])}</i>",
            permanent=False,
        ),
        icon=folium.Icon(color="red", icon="home", prefix="fa"),
    ).add_to(fmap)

    # ── Delivery stop base markers ─────────────────────────────────────────────
    for node_idx, pt in idx_to_pt.items():
        safe_name = html_mod.escape(pt.name)
        folium.CircleMarker(
            location=[pt.lat, pt.lon],
            radius=8,
            color="steelblue",
            weight=2,
            fill=True,
            fill_color="steelblue",
            fill_opacity=0.55,
            tooltip=folium.Tooltip(
                f"<b>{safe_name}</b><br>Node #{node_idx} | Demand: {pt.demand} units",
                permanent=False,
            ),
        ).add_to(fmap)

    # ── Route polylines + sequence badges ─────────────────────────────────────
    depot_ll = [depot["lat"], depot["lon"]]

    for v_idx, (route, r_dist) in enumerate(
        zip(solution["routes"], solution["route_distances"])
    ):
        color       = colors[v_idx % len(colors)]
        stops_only  = [n for n in route if n != 0]
        n_stops     = len(stops_only)

        # Build ordered [lat, lon] coordinate sequence for this route
        coord_seq: List[List[float]] = []
        for node in route:
            if node == 0:
                coord_seq.append(depot_ll)
            elif node in idx_to_pt:
                p = idx_to_pt[node]
                coord_seq.append([p.lat, p.lon])

        # v_idx, n_stops, r_dist are all numeric — no escaping needed
        tooltip_text = f"Vehicle {v_idx + 1} | {n_stops} stops | {r_dist:.1f} km"

        if _ANTPATH_AVAILABLE:
            AntPath(
                locations=coord_seq,
                color=color,
                weight=4,
                opacity=0.85,
                delay=900,
                tooltip=tooltip_text,
            ).add_to(fmap)
        else:
            folium.PolyLine(
                locations=coord_seq,
                color=color,
                weight=4,
                opacity=0.80,
                tooltip=tooltip_text,
            ).add_to(fmap)

        # Sequence number badge at each delivery stop
        for seq_pos, node in enumerate(stops_only):
            if node not in idx_to_pt:
                continue
            pt = idx_to_pt[node]
            badge_html = (
                f'<div style="'
                f'font-size:9px;font-weight:bold;color:{color};'
                f'background:white;border:2px solid {color};'
                f'border-radius:50%;width:20px;height:20px;'
                f'text-align:center;line-height:16px;'
                f'box-shadow:1px 1px 3px rgba(0,0,0,0.35);'
                f'">{seq_pos + 1}</div>'
            )
            folium.Marker(
                location=[pt.lat, pt.lon],
                icon=folium.DivIcon(
                    html=badge_html,
                    icon_size=(20, 20),
                    icon_anchor=(10, 10),
                ),
            ).add_to(fmap)

    # ── Floating legend ────────────────────────────────────────────────────────
    legend_rows = "".join(
        f'<tr>'
        f'<td><span style="display:inline-block;width:14px;height:14px;'
        f'background:{colors[i % len(colors)]};border-radius:3px;'
        f'vertical-align:middle;"></span></td>'
        f'<td style="padding-left:6px;">Vehicle {i + 1}</td>'
        f'<td style="padding-left:10px;color:#555;">'
        f'{len([n for n in solution["routes"][i] if n != 0])} stops &nbsp;|&nbsp; '
        f'{solution["route_distances"][i]:.1f} km</td>'
        f'</tr>'
        for i in range(len(solution["routes"]))
    )
    total_stops = sum(len([n for n in r if n != 0]) for r in solution["routes"])
    legend_html = f"""
    <div style="
        position:fixed;bottom:24px;left:24px;z-index:9999;
        background:rgba(255,255,255,0.96);
        padding:12px 16px;border-radius:10px;
        box-shadow:2px 2px 10px rgba(0,0,0,0.25);
        font-family:Arial,sans-serif;font-size:12px;min-width:240px;">
      <b style="font-size:13px;">Optimised Route Summary</b><br>
      <span style="color:#666;font-size:11px;">
        {total_stops} stops &nbsp;·&nbsp; {solution['total_distance']:.1f} km total
        &nbsp;·&nbsp; {len(solution['routes'])} vehicles
      </span>
      <table style="margin-top:8px;border-collapse:collapse;">{legend_rows}</table>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))

    return fmap


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — CSV EXPORT
# ─────────────────────────────────────────────────────────────────────────────

_CSV_INJECTION_CHARS = frozenset("=+-@\t\r\n|")


def _sanitize_csv_field(value: str) -> str:
    """
    Prevent CSV formula injection (CWE-1236).
    Excel / LibreOffice treat cells starting with =, +, -, @, tab, or CR/LF
    as formulas. Prefix them with a single-quote so they are treated as text.
    """
    if value and value[0] in _CSV_INJECTION_CHARS:
        return "'" + value
    return value


def export_csv(
    solution: Dict,
    depot:    Dict,
    active:   List[GeoPoint],
    path:     Path,
) -> pd.DataFrame:
    """
    Write the full stop sequence to a CSV with columns:
        stop_sequence, assigned_vehicle_id, node_index, stop_name,
        address, latitude, longitude, demand,
        arrival_distance_milestone_km
    """
    # Internal path-traversal guard — safe even if called directly, not via main()
    safe_path = _safe_output_path(path, OUTPUT_DIR.resolve())

    idx_to_pt = _node_map(active)
    rows: List[Dict] = []

    # Defined once outside the loop — creating it per-iteration was wasteful
    # and obscured that depot/idx_to_pt are captured by reference, not by value.
    def _coords(n: int) -> Tuple[float, float]:
        if n == 0:
            return depot["lat"], depot["lon"]
        return idx_to_pt[n].lat, idx_to_pt[n].lon

    for v_idx, route in enumerate(solution["routes"]):
        cum_km = 0.0
        for step, node in enumerate(route):
            if step > 0:
                prev = route[step - 1]
                cum_km += haversine_km(*_coords(prev), *_coords(node))

            if node == 0:
                name, addr   = depot["name"], depot["address"]
                lat, lon     = depot["lat"], depot["lon"]
                demand       = 0
            else:
                pt           = idx_to_pt[node]
                name, addr   = pt.name, pt.address
                lat, lon     = pt.lat, pt.lon
                demand       = pt.demand

            rows.append({
                "stop_sequence":                 step,
                "assigned_vehicle_id":           f"V{v_idx + 1}",
                "node_index":                    node,
                "stop_name":                     _sanitize_csv_field(name),
                "address":                       _sanitize_csv_field(addr),
                "latitude":                      round(lat, 6),
                "longitude":                     round(lon, 6),
                "demand":                        demand,
                "arrival_distance_milestone_km": round(cum_km, 3),
            })

    df = pd.DataFrame(rows)
    _safe_write_check(safe_path)
    df.to_csv(safe_path, index=False)
    log.info("Stop sequence exported → %s  (%d rows)", safe_path, len(df))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — CONSOLE METRICS
# ─────────────────────────────────────────────────────────────────────────────

def print_metrics(
    solution:         Dict,
    active:           List[GeoPoint],
    runtime_sec:      float,
    num_vehicles:     int,
    vehicle_capacity: int,
) -> None:
    """Print a formatted performance summary to stdout."""
    idx_to_pt   = _node_map(active)
    n_routes    = len(solution["routes"])
    avg_km      = solution["total_distance"] / n_routes if n_routes else 0.0
    total_demand = sum(p.demand for p in active)

    vehicle_loads = [
        sum(idx_to_pt[n].demand for n in route if n != 0 and n in idx_to_pt)
        for route in solution["routes"]
    ]
    avg_cap_util = (
        np.mean([ld / vehicle_capacity * 100.0 for ld in vehicle_loads])
        if vehicle_loads and vehicle_capacity > 0 else 0.0
    )

    bar = "─" * 58
    solver_tag = "OR-Tools GLS (CVRP)" if ORTOOLS_AVAILABLE else "Nearest-Neighbour + 2-opt"

    print(f"\n{bar}")
    print("  ROUTE OPTIMISATION SUMMARY")
    print(bar)
    print(f"  Solver            :  {solver_tag}")
    print(f"  Status            :  {solution['status']}")
    print(f"  Wall-clock time   :  {runtime_sec:.2f} s")
    print(bar)
    print(f"  Stops geocoded    :  {len(active)} / {len(active)}")
    print(f"  Vehicles used     :  {n_routes} / {num_vehicles}")
    print(f"  Total demand      :  {total_demand} units")
    print(f"  Vehicle capacity  :  {vehicle_capacity} units each")
    print(f"  Avg cap. util.    :  {avg_cap_util:.1f} %")
    print(bar)
    print(f"  Total distance    :  {solution['total_distance']:.2f} km")
    print(f"  Avg / route       :  {avg_km:.2f} km")
    print(bar)
    for v_idx, (route, r_dist) in enumerate(
        zip(solution["routes"], solution["route_distances"])
    ):
        stops = [n for n in route if n != 0]
        load  = sum(idx_to_pt[n].demand for n in stops if n in idx_to_pt)
        util  = (load / vehicle_capacity * 100.0) if vehicle_capacity > 0 else 0.0
        print(
            f"  V{v_idx + 1}:  {len(stops):2d} stops  |  "
            f"{r_dist:6.2f} km  |  load {load:3d}/{vehicle_capacity}  ({util:.0f}%)"
        )
    print(bar)
    print(f"  Map  → {MAP_HTML}")
    print(f"  CSV  → {STOPS_CSV}")
    print(bar)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — MAIN ORCHESTRATION
# ─────────────────────────────────────────────────────────────────────────────

def _validate_config() -> None:
    """
    Validate all module-level configuration values before any I/O or computation.
    Raises ValueError / TypeError on the first invalid setting found.
    Catches misconfiguration that would otherwise cause silent bad math,
    division-by-zero, or solver hangs.
    """
    # ── Depot coordinates ──────────────────────────────────────────────────────
    for key, lo, hi in [("lat", -90.0, 90.0), ("lon", -180.0, 180.0)]:
        val = DEPOT.get(key)
        if not isinstance(val, (int, float)):
            raise TypeError(f"DEPOT['{key}'] must be a number, got {type(val).__name__}")
        if not math.isfinite(val):
            raise ValueError(f"DEPOT['{key}'] = {val} is not a finite number (NaN or Inf)")
        if not (lo <= val <= hi):
            raise ValueError(f"DEPOT['{key}'] = {val} is outside valid range [{lo}, {hi}]")

    # ── Fleet parameters ───────────────────────────────────────────────────────
    # bool subclass check required: isinstance(True, int) is True in Python,
    # so True/False would silently become 1/0 without the explicit bool guard.
    if not isinstance(NUM_VEHICLES, int) or isinstance(NUM_VEHICLES, bool) or NUM_VEHICLES <= 0:
        raise ValueError(f"NUM_VEHICLES must be a positive integer (got {NUM_VEHICLES!r})")
    if not isinstance(VEHICLE_CAPACITY, int) or isinstance(VEHICLE_CAPACITY, bool) or VEHICLE_CAPACITY <= 0:
        raise ValueError(f"VEHICLE_CAPACITY must be a positive integer (got {VEHICLE_CAPACITY!r})")

    _MAX_ROUTE_KM_LIMIT = 9.2e15   # OR-Tools int64 ceiling: 2^63-1 / SCALE(100)
    if (not isinstance(MAX_ROUTE_KM, (int, float))
            or not math.isfinite(MAX_ROUTE_KM)
            or MAX_ROUTE_KM < 0):
        raise ValueError(f"MAX_ROUTE_KM must be a non-negative finite number (got {MAX_ROUTE_KM!r})")
    if MAX_ROUTE_KM > _MAX_ROUTE_KM_LIMIT:
        raise ValueError(
            f"MAX_ROUTE_KM = {MAX_ROUTE_KM:.2e} exceeds OR-Tools int64 ceiling "
            f"({_MAX_ROUTE_KM_LIMIT:.2e} km). Use a realistic value."
        )

    # ── Solver parameters ──────────────────────────────────────────────────────
    if not isinstance(SOLVER_TIME_SEC, int) or SOLVER_TIME_SEC < 1:
        raise ValueError(
            f"SOLVER_TIME_SEC must be a positive integer >= 1 (got {SOLVER_TIME_SEC!r}). "
            "A zero time limit would skip Guided Local Search entirely."
        )

    if MAX_ROUTE_KM == 0:
        log.info("MAX_ROUTE_KM = 0: per-vehicle distance constraint is disabled.")


def _safe_output_path(path: Path, base: Path) -> Path:
    """
    Resolve *path* and assert it lives inside *base* (path-traversal guard).
    Raises ValueError if the resolved path escapes the base directory.
    """
    resolved      = path.resolve()
    resolved_base = base.resolve()
    try:
        resolved.relative_to(resolved_base)
    except ValueError:
        raise ValueError(
            f"Output path '{path}' resolves outside the allowed base "
            f"directory '{resolved_base}'. Check OUTPUT_DIR configuration."
        )
    return resolved


def _safe_write_check(path: Path) -> None:
    """
    Assert the path is not a symlink immediately before writing.
    Guards against TOCTOU attacks: an adversary replacing vrp_output/ with
    a symlink during the ~30–75 s solver window to redirect writes to
    arbitrary locations (e.g. /etc/passwd on a shared system).
    """
    if path.is_symlink():
        raise ValueError(
            f"Output path '{path}' is a symbolic link. "
            "This may indicate a TOCTOU race-condition attack. Aborting."
        )
    # Also guard the parent directory
    if path.parent.is_symlink():
        raise ValueError(
            f"Output directory '{path.parent}' is a symbolic link. Aborting."
        )


def main() -> Optional[Dict]:
    t0 = time.perf_counter()

    # ── Pre-flight: validate config and inputs before any I/O ─────────────────
    _validate_config()                 # depot coords, fleet params, solver settings
    validate_raw_stops(RAW_STOPS)      # stop structure, types, bounds

    # ── Validate and resolve output paths ─────────────────────────────────────
    safe_output_dir = _safe_output_path(OUTPUT_DIR, Path.cwd())
    safe_map_html   = _safe_output_path(MAP_HTML,   Path.cwd())
    safe_stops_csv  = _safe_output_path(STOPS_CSV,  Path.cwd())

    # mkdir only after all validation passes — avoids ghost directories on failure
    safe_output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Geocode all stops ──────────────────────────────────────────────
    log.info("=== STEP 1/5  Geocoding %d stops ===", len(RAW_STOPS))
    all_points = geocode_stops(RAW_STOPS)
    active     = [p for p in all_points if p.geocoded]

    if len(active) < 2:
        log.error("Fewer than 2 stops geocoded successfully. Aborting.")
        return None

    log.info("Geocoding complete: %d/%d stops resolved.", len(active), len(all_points))

    # ── Step 2: Build distance matrix ─────────────────────────────────────────
    log.info("=== STEP 2/5  Building %d×%d distance matrix ===",
             len(active) + 1, len(active) + 1)

    all_coords: List[Tuple[float, float]] = (
        [(DEPOT["lat"], DEPOT["lon"])]
        + [(p.lat, p.lon) for p in active]
    )
    demands = [0] + [p.demand for p in active]   # index 0 = depot (demand 0)

    dist_matrix = build_distance_matrix(
        all_coords,
        use_road_network=USE_ROAD_NETWORK,
        osrm_url=OSRM_BASE_URL,
    )
    log.info("Distance matrix ready.  Max element: %.2f km", dist_matrix.max())

    # ── Step 3: Optimise ───────────────────────────────────────────────────────
    log.info("=== STEP 3/5  Solving CVRP  (%s) ===",
             "OR-Tools" if ORTOOLS_AVAILABLE else "heuristic")

    solution: Optional[Dict] = None

    if ORTOOLS_AVAILABLE:
        solution = solve_cvrp_ortools(
            dist_matrix=dist_matrix,
            demands=demands,
            num_vehicles=NUM_VEHICLES,
            vehicle_capacity=VEHICLE_CAPACITY,
            max_route_km=MAX_ROUTE_KM,
            time_limit_sec=SOLVER_TIME_SEC,
        )

    if solution is None:
        log.info("Activating nearest-neighbour fallback...")
        solution = solve_cvrp_nearest_neighbor(
            dist_matrix=dist_matrix,
            demands=demands,
            num_vehicles=NUM_VEHICLES,
            vehicle_capacity=VEHICLE_CAPACITY,
            max_route_km=MAX_ROUTE_KM,
        )

    if not solution["routes"]:
        log.error(
            "No routes generated. Constraints may be infeasible "
            "(try increasing VEHICLE_CAPACITY or MAX_ROUTE_KM)."
        )
        return None

    # ── Step 4: Build and save the interactive map ─────────────────────────────
    log.info("=== STEP 4/5  Rendering Folium map ===")
    fmap = build_folium_map(DEPOT, active, solution, ROUTE_COLORS)
    _safe_write_check(safe_map_html)
    fmap.save(str(safe_map_html))
    log.info("Map saved → %s", safe_map_html)

    # ── Step 5: Export CSV and print metrics ───────────────────────────────────
    log.info("=== STEP 5/5  Exporting results ===")
    df = export_csv(solution, DEPOT, active, safe_stops_csv)

    t1 = time.perf_counter()
    print_metrics(solution, active, t1 - t0, NUM_VEHICLES, VEHICLE_CAPACITY)

    return {"solution": solution, "dataframe": df}


if __name__ == "__main__":
    main()
