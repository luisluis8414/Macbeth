"""Maps that look like real generals.io free-for-all maps.

The simulator's generator (generals-bots) differs from the real game in ways a policy can learn and then misuse
online (measured on 1,148 real 8-player replays of top players against the generator at 8 players, sizes 25-32):

    real                                         generator
    sides 25-33, mostly 25-27 (median 27)        uniform; after the spawn filter mostly large (median 30)
    generals anywhere (median 3 from the edge)   at the edge (median 1)
    every general has a city within 6 steps,     75 % do
      30 % right next to it
    17-32 cities (median 24)                     18-20
    no spawn pattern known                       every general gets about as much room within 7 steps as the
                                                 first one, which narrows down where enemy generals can be

Generator terrain (mountains, cities) is kept. realistic_pool replaces the spawns and the number of cities with the
real distributions and draws the map sizes like the replays. The real numbers are below; scripts/data_report.py
measures the replays they came from.
"""
import numpy as np
from scipy import ndimage

# side lengths of real 8-player maps (rows and columns drawn independently: 15 % of real maps are square, which is
# what independent draws give)
REAL_SIDES = {25: 553, 26: 406, 27: 280, 28: 261, 29: 221, 30: 185, 31: 169, 32: 144, 33: 77}
# number of cities on a real 8-player map
REAL_CITIES = {17: 10, 18: 26, 19: 32, 20: 64, 21: 47, 22: 109, 23: 134, 24: 169, 25: 148, 26: 110, 27: 102, 28: 69,
               29: 56, 30: 37, 31: 27, 32: 6}
# walking distance from a general to its nearest city (cities count as walkable)
REAL_NEAREST_CITY = {1: 973, 2: 785, 3: 562, 4: 402, 5: 322, 6: 144}
MIN_MANHATTAN = 10              # no two real generals are closer than 10 rows + columns
CITY_ARMY = (40, 51)            # neutral city armies, as in the simulator and the replays (40-50)


def _draw(rng, table: dict) -> int:
    keys = np.array(list(table))
    p = np.array(list(table.values()), float)
    return int(rng.choice(keys, p=p / p.sum()))


def _distances(passable: np.ndarray, start: tuple) -> np.ndarray:
    """Walking distance from start to every cell (4-neighbours over passable cells), -1 where unreachable."""
    dist = np.full(passable.shape, -1)
    frontier = np.zeros_like(passable)
    frontier[start] = True
    seen, d = frontier.copy(), 0
    while frontier.any():
        dist[frontier] = d
        grown = ndimage.binary_dilation(frontier) & passable & ~seen
        seen |= grown
        frontier, d = grown, d + 1
    return dist


def map_size(mountains: np.ndarray) -> tuple[int, int]:
    """Rows and columns a map uses (maps start at the top left; the rest is padding mountains)."""
    open_ = ~mountains
    return int(np.flatnonzero(open_.any(1))[-1]) + 1, int(np.flatnonzero(open_.any(0))[-1]) + 1


def realistic_map(state: dict, n_players: int, rng: np.random.Generator) -> dict | None:
    """One map (numpy fields of a GameState at time 0) with real spawns and city counts, or None if the terrain has
    no room for them. Mountains stay; generals, ownership, cities and armies are redone."""
    mountains = state["mountains"]
    h, w = map_size(mountains)
    castles = state["castles"].copy()
    armies = np.where(castles, state["armies"], 0)
    passable = ~mountains
    labels, _ = ndimage.label(passable)
    main = labels == np.bincount(labels[passable]).argmax()            # the largest connected area

    # generals: uniformly on open ground (no city) of the main area, at least MIN_MANHATTAN apart
    cells = np.argwhere(main & ~castles)
    for _ in range(20):
        order = rng.permutation(len(cells))
        picked = []
        for i in order:
            r, c = cells[i]
            if all(abs(r - pr) + abs(c - pc) >= MIN_MANHATTAN for pr, pc in picked):
                picked.append((int(r), int(c)))
                if len(picked) == n_players:
                    break
        if len(picked) == n_players:
            break
    else:
        return None
    occupied = np.zeros_like(passable)
    for p in picked:
        occupied[p] = True

    # a city near every general, its distance drawn like the real nearest-city distances
    keep = np.zeros_like(passable)                                      # each general's nearest city
    for p in picked:
        dist = _distances(passable, p)
        want = _draw(rng, REAL_NEAREST_CITY)
        reach = castles & (dist >= 0)
        if reach.any() and dist[reach].min() <= want:
            near = np.argwhere(reach & (dist == dist[reach].min()))[0]
        else:
            spots = np.argwhere((dist == want) & ~castles & ~occupied)
            if len(spots) == 0:
                spots = np.argwhere((dist > 0) & (dist <= 6) & ~castles & ~occupied)
            if len(spots) == 0:
                continue
            near = spots[rng.integers(len(spots))]
            castles[tuple(near)], armies[tuple(near)] = True, rng.integers(*CITY_ARMY)
        keep[tuple(near)] = True

    # then a real number of cities: more anywhere on open ground (the generator stops at about 20), or fewer by
    # dropping cities that are no general's nearest
    target = _draw(rng, REAL_CITIES)
    free = np.argwhere(passable & ~castles & ~occupied)
    extra = max(0, min(target - int(castles.sum()), len(free)))
    for q in free[rng.choice(len(free), extra, replace=False)]:
        castles[tuple(q)], armies[tuple(q)] = True, rng.integers(*CITY_ARMY)
    spare = np.argwhere(castles & ~keep)
    surplus = max(0, min(int(castles.sum()) - target, len(spare)))
    for q in spare[rng.choice(len(spare), surplus, replace=False)]:
        castles[tuple(q)], armies[tuple(q)] = False, 0

    ownership = np.zeros((n_players,) + passable.shape, bool)
    generals = np.zeros_like(passable)
    for k, p in enumerate(picked):
        ownership[(k,) + p] = True
        generals[p] = True
        armies[p] = 1
    return dict(state, armies=armies.astype(state["armies"].dtype), ownership=ownership,
                ownership_neutral=passable & ~ownership.any(0), generals=generals, castles=castles,
                general_positions=np.array(picked, dtype=state["general_positions"].dtype))


def realistic_pool(pools: list, size: int, n_players: int, rng: np.random.Generator):
    """A pool of `size` maps (GameState, batched) from generator pools: map sizes drawn like real ones, then
    realistic_map on each. pools: GameStates from GeneralsEnv.reset with sides covering 25-33."""
    import jax
    import jax.numpy as jnp
    states = [jax.tree.map(lambda x, i=i: np.asarray(x[i]), p) for p in pools for i in range(p.armies.shape[0])]
    sides = [map_size(np.asarray(s.mountains)) for s in states]
    p_side = {s: n / sum(REAL_SIDES.values()) for s, n in REAL_SIDES.items()}
    per_combo = {}
    for hw in sides:
        per_combo[hw] = per_combo.get(hw, 0) + 1
    weight = np.array([p_side.get(r, 0) * p_side.get(c, 0) / per_combo[(r, c)] for r, c in sides])
    out = []
    while len(out) < size:
        i = rng.choice(len(states), p=weight / weight.sum())
        m = realistic_map(states[i]._asdict(), n_players, rng)
        if m is not None:
            out.append(type(states[i])(**m))
    return jax.tree.map(lambda *xs: jnp.asarray(np.stack(xs)), *out)
