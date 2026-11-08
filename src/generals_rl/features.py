"""Agent observation (computed from the full GameState, FFA-ready) and the flat action space.

Observation for player `me`, a dict of arrays:
    planes  (H, W, len(PLANE_NAMES)) per-cell features under fog of war (82 planes)
    players (MAX_PLAYERS, len(PLAYER_FEATURE_NAMES))  one row per player slot: public scoreboard + its recent changes
    player_mask (MAX_PLAYERS,)        True for slots of players still in the game
    action_mask (H*W*8 + 1,)          legal actions

Player slots are relative to the viewer: slot 0 is `me`, slot k is player (me + k) mod N. The order is fixed
for a whole game, so "slot 2" always means the same opponent. Games with fewer than 8 players leave slots empty.

Besides the current board, a player keeps what it has seen (the memory in the history): where it saw generals,
cities and mountains, each opponent's land when last seen, enemy armies and how long ago, and a fading trail of its
own moves. From that come walking distances to its own general, to each remembered enemy general and to its land.

Action space (per board of size H x W):
    index = ((row * W + col) * 4 + direction) * 2 + split     for moves
    index = H * W * 8                                         for pass
Simulator action format (generals-bots): [pass, row, col, direction, split],
direction 0=UP 1=DOWN 2=LEFT 3=RIGHT, split 1 = move half.
"""
import jax
import jax.numpy as jnp

from generals.core.action import compute_valid_move_mask
from generals.core.game import GameState, get_visibility

NUM_DIRECTIONS = 4
NUM_SPLITS = 2
MAX_PLAYERS = 8            # generals.io FFA has up to 8 players
HISTORY_TICKS = 32         # past ticks in the move-history planes (weights 1, 31/32, ..., 1/32)
SCORE_LAGS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)   # scoreboard changes over these many ticks
SCORE_HISTORY = max(SCORE_LAGS)
ARMY_DELTAS = 7            # per-cell army changes over this many ticks
BFS_STEPS = 64             # walking distances up to this many cells
TRAIL_DECAY = 0.96         # the own trail fades by this factor per tick
PASS = (1, 0, 0, 0, 0)
DIRECTIONS = jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]], jnp.int32)

_OPP_SLOTS = range(1, MAX_PLAYERS)
PLANE_NAMES = (
    "army_own", "army_enemy", "army_neutral",           # army / 50, visible cells only
    "own", "neutral",                                   # visible ownership
    "general", "castle", "mountain",                    # visible structures
    "fog", "structure_in_fog",                          # a player sees obstacles in the fog, not their kind
) + tuple(f"slot_{k}" for k in _OPP_SLOTS) + (         # which opponent owns a visible cell
    # history of the last HISTORY_TICKS ticks, newer ticks weigh more:
    "own_move_from", "own_move_to",                     # my own recent moves
) + tuple(f"enemy_move_from_{k}" for k in _OPP_SLOTS) + tuple(f"enemy_move_to_{k}" for k in _OPP_SLOTS) + (
    "enemy_army_recent",                                # enemy army / 50 where I saw enemy troops recently
    # memory, kept per game outside the network:
    "known_enemy_general", "known_city", "known_mountain",  # ever seen (a general until its cell is seen without it)
    "ever_seen", "enemy_ever_seen",
    "last_enemy_army", "last_enemy_army_age",           # last seen count / 50, and log(1 + ticks since) / 5
) + tuple(f"own_army_delta_{k}" for k in range(1, ARMY_DELTAS + 1)) + tuple(
    f"enemy_army_delta_{k}" for k in range(1, ARMY_DELTAS + 1)) + (
    "coord_row", "coord_col",                           # from the map's own corner, 0 off the map
) + tuple(f"known_general_{k}" for k in _OPP_SLOTS) + tuple(f"last_land_{k}" for k in _OPP_SLOTS) + (
    "dist_own_general",) + tuple(f"dist_general_{k}" for k in _OPP_SLOTS) + ("dist_own_land", "own_trail", "on_board")

PLAYER_FEATURE_NAMES = ("is_me", "alive", "log_land", "log_army", "land_share", "army_share", "time",
                        "time_mod_50") + tuple(f"d_{q}_{lag}" for lag in SCORE_LAGS for q in ("land", "army")) + (
                        "time_odd",)        # signed log changes vs. lag ticks ago; parity: growth and move order


def num_actions(h: int, w: int) -> int:
    return h * w * NUM_DIRECTIONS * NUM_SPLITS + 1


def encode_action(action: jnp.ndarray, w: int, h: int) -> jnp.ndarray:
    """[pass, row, col, dir, split] (..., 5) -> flat index (...)."""
    is_pass, r, c, d, s = (action[..., i] for i in range(5))
    move_idx = ((r * w + c) * NUM_DIRECTIONS + d) * NUM_SPLITS + s
    return jnp.where(is_pass == 1, h * w * NUM_DIRECTIONS * NUM_SPLITS, move_idx)


def decode_action(index: jnp.ndarray, w: int, h: int) -> jnp.ndarray:
    """Flat index (...) -> [pass, row, col, dir, split] (..., 5)."""
    pass_idx = h * w * NUM_DIRECTIONS * NUM_SPLITS
    is_pass = index == pass_idx
    idx = jnp.where(is_pass, 0, index)
    s = idx % NUM_SPLITS
    d = (idx // NUM_SPLITS) % NUM_DIRECTIONS
    cell = idx // (NUM_SPLITS * NUM_DIRECTIONS)
    r, c = cell // w, cell % w
    return jnp.stack([is_pass.astype(jnp.int32), r, c, d, s], axis=-1).astype(jnp.int32)


def action_mask(state: GameState, me) -> jnp.ndarray:
    """Legal actions of player `me` (own cell with army > 1, target on the board and not a mountain).
    Pass is always legal."""
    moves = compute_valid_move_mask(state.armies, state.ownership[me], state.mountains)  # (H, W, 4)
    moves = jnp.repeat(moves[..., None], NUM_SPLITS, axis=-1)                           # (H, W, 4, 2)
    return jnp.concatenate([moves.reshape(-1), jnp.ones((1,), dtype=bool)])


def scoreboard(state: GameState) -> jnp.ndarray:
    """(N, 2) land and army of every player: the public scoreboard."""
    land = state.ownership.sum((1, 2))
    army = (state.armies[None] * state.ownership).sum((1, 2))
    return jnp.stack([land, army], -1).astype(jnp.int32)


def _army_plane(x):
    """Army counts on the board planes: count / 50, so one unit looks the same at every army size."""
    return jnp.maximum(x.astype(jnp.float32), 0.0) / 50.0


def history_init(state: GameState) -> dict:
    """History at the start of a game: no moves yet, the scoreboard has not changed, nothing remembered.

    A history holds what happened before the current tick t, newest first:
        states  GameState stacked (HISTORY_TICKS, ...)   states at t-1, t-2, ...
        actions (HISTORY_TICKS, N, 5)                    the actions taken in those states
        score   (SCORE_HISTORY, N, 2)                    scoreboard at t-1, t-2, ...
        memory  what every player has seen so far (_remember), (N, ...) per field
    """
    n = state.ownership.shape[0]
    empty = jnp.zeros_like(state.ownership)
    by_owner = jnp.zeros((n,) + state.ownership.shape, bool)                  # (viewer, owner, H, W)
    return dict(
        states=jax.tree.map(lambda x: jnp.broadcast_to(x, (HISTORY_TICKS,) + x.shape), state),
        actions=jnp.broadcast_to(jnp.array(PASS, jnp.int32), (HISTORY_TICKS, n, 5)),
        score=jnp.broadcast_to(scoreboard(state), (SCORE_HISTORY, n, 2)),
        memory=dict(general=empty, city=empty, mountain=empty, seen=empty, enemy_seen=empty,
                    army=jnp.zeros_like(state.ownership, jnp.float32), age=jnp.zeros_like(state.ownership, jnp.float32),
                    general_by=by_owner, land_by=by_owner, trail=jnp.zeros_like(state.ownership, jnp.float32)),
    )


def _remember(memory: dict, state: GameState) -> dict:
    """Memory of every player (N, H, W) after looking at `state`. Only what that player can see gets in."""
    visible = get_visibility(state.ownership)
    enemy = state.ownership.any(0, keepdims=True) & ~state.ownership & visible
    n = state.ownership.shape[0]
    vis, own = visible[:, None], state.ownership[None] & ~jnp.eye(n, dtype=bool)[:, :, None, None]
    return dict(
        # per owner, as each viewer saw it
        general_by=(memory["general_by"] & ~vis) | (state.generals & own & vis),
        land_by=jnp.where(vis, own, memory["land_by"]),
        trail=memory["trail"],
        # a remembered general is dropped once its cell is seen without one (in FFA: captured, the game goes on)
        general=(memory["general"] & ~visible) | (state.generals[None] & enemy),
        city=memory["city"] | (state.castles[None] & visible),
        mountain=memory["mountain"] | (state.mountains[None] & visible),
        seen=memory["seen"] | visible,
        enemy_seen=memory["enemy_seen"] | enemy,
        army=jnp.where(enemy, state.armies[None].astype(jnp.float32), memory["army"]),
        age=jnp.where(enemy, 0.0, memory["age"] + 1.0),
    )


def history_push(hist: dict, state: GameState, actions: jnp.ndarray) -> dict:
    """History after playing `actions` (N, 5) in `state` (the state before the step)."""
    push = lambda buf, x: jnp.concatenate([x[None], buf[:-1]])
    memory = _remember(hist["memory"], state)
    memory["trail"] = jnp.maximum(hist["memory"]["trail"] * TRAIL_DECAY, _move_cells(actions, *state.armies.shape))
    return dict(states=jax.tree.map(push, hist["states"], state), actions=push(hist["actions"], actions),
                score=push(hist["score"], scoreboard(state)), memory=memory)


def _move_cells(actions: jnp.ndarray, h: int, w: int) -> jnp.ndarray:
    """(N, H, W) 1.0 on the source and target cell of each player's move (nothing for a pass)."""
    moving = actions[:, 0] == 0
    r, c = actions[:, 1], actions[:, 2]
    r2 = jnp.clip(r + DIRECTIONS[actions[:, 3], 0], 0, h - 1)
    c2 = jnp.clip(c + DIRECTIONS[actions[:, 3], 1], 0, w - 1)
    rr, cc = jnp.arange(h)[None, :, None], jnp.arange(w)[None, None, :]
    at = lambda y, x: (rr == y[:, None, None]) & (cc == x[:, None, None])
    return ((at(r, c) | at(r2, c2)) & moving[:, None, None]).astype(jnp.float32)


def walking_distance(sources: jnp.ndarray, walls: jnp.ndarray) -> jnp.ndarray:
    """(K, H, W) steps from the nearest source cell of each of the K source maps, moving between non-wall cells,
    scaled to [0, 1]: BFS_STEPS or more (or unreachable) is 1."""
    far = jnp.int16(BFS_STEPS)
    d = jnp.where(sources, jnp.int16(0), far)

    def relax(_, d):
        p = jnp.pad(d, ((0, 0), (1, 1), (1, 1)), constant_values=far)
        near = jnp.minimum(jnp.minimum(p[:, :-2, 1:-1], p[:, 2:, 1:-1]), jnp.minimum(p[:, 1:-1, :-2], p[:, 1:-1, 2:]))
        return jnp.where(walls & ~sources, far, jnp.minimum(d, near + 1))
    d = jax.lax.fori_loop(0, BFS_STEPS, relax, d)
    return jnp.minimum(d, far).astype(jnp.float32) / BFS_STEPS


def map_extent(state: GameState):
    """First and last row and column of the map inside the padding (the padding is all mountains), plus the mask."""
    open_ = ~state.mountains
    rows, cols = open_.any(1), open_.any(0)
    h, w = rows.shape[0], cols.shape[0]
    r0, r1 = jnp.argmax(rows), h - 1 - jnp.argmax(rows[::-1])
    c0, c1 = jnp.argmax(cols), w - 1 - jnp.argmax(cols[::-1])
    rr, cc = jnp.arange(h)[:, None], jnp.arange(w)[None, :]
    return r0, r1, c0, c1, (rr >= r0) & (rr <= r1) & (cc >= c0) & (cc <= c1)


def _memory_planes(hist: dict, state: GameState, me, f, full: bool = False) -> list:
    """What `me` remembers, per-cell army changes (newest first), coordinates, walking distances and the trail.
    full: no fog, so memory is the whole current state (for the privileged critic)."""
    memory = jax.tree.map(lambda x: x[me], _remember(hist["memory"], state))
    n = state.ownership.shape[0]
    if full:
        others = state.ownership & (jnp.arange(n) != me)[:, None, None]
        enemy_now = others.any(0)
        memory = dict(memory, general=state.generals & enemy_now, city=state.castles, mountain=state.mountains,
                      seen=jnp.ones_like(enemy_now), enemy_seen=enemy_now,
                      army=jnp.where(enemy_now, state.armies, 0).astype(jnp.float32), age=jnp.zeros_like(memory["age"]),
                      general_by=state.generals & others, land_by=others)

    def armies(st):
        vis = jnp.ones_like(st.mountains) if full else get_visibility(st.ownership[me])
        enemy = st.ownership.any(0) & ~st.ownership[me] & vis
        return st.armies * st.ownership[me], st.armies * enemy

    past_own, past_enemy = jax.vmap(armies)(hist["states"])                     # newest first
    now_own, now_enemy = armies(state)
    own = f(jnp.concatenate([now_own[None], past_own]))
    enemy = f(jnp.concatenate([now_enemy[None], past_enemy]))
    h, w = state.armies.shape
    r0, r1, c0, c1, on_board = map_extent(state)
    coord_row = jnp.clip(f(jnp.arange(h) - r0) / jnp.maximum(r1 - r0, 1), 0.0, 1.0)[:, None] * on_board
    coord_col = jnp.clip(f(jnp.arange(w) - c0) / jnp.maximum(c1 - c0, 1), 0.0, 1.0)[None, :] * on_board

    # Per opponent slot: remembered general and last seen land. Then walking distances: obstacles are every mountain
    # and every structure in the fog not known to be a city (a player sees obstacles through the fog, not their kind).
    slot = lambda k: (me + k) % n
    known_general = [memory["general_by"][slot(k)] if k < n else jnp.zeros((h, w), bool) for k in _OPP_SLOTS]
    last_land = [memory["land_by"][slot(k)] if k < n else jnp.zeros((h, w), bool) for k in _OPP_SLOTS]
    walls = state.mountains | (state.castles & ~memory["city"])
    mine = state.ownership[me]
    sources = jnp.stack([state.generals & mine] + known_general[:n - 1] + [mine])       # opponents that exist only
    dist = walking_distance(sources, walls)
    dist_general = [dist[k] if k < n else jnp.ones((h, w), jnp.float32) for k in _OPP_SLOTS]
    return [
        f(memory["general"]), f(memory["city"]), f(memory["mountain"]), f(memory["seen"]), f(memory["enemy_seen"]),
        memory["army"] / 50.0, jnp.log1p(memory["age"]) / 5.0,
        *[(own[k] - own[k + 1]) / 50.0 for k in range(ARMY_DELTAS)],
        *[(enemy[k] - enemy[k + 1]) / 50.0 for k in range(ARMY_DELTAS)],
        coord_row, coord_col,
        *[f(x) for x in known_general], *[f(x) for x in last_land],
        dist[0], *dist_general, dist[n], memory["trail"], f(on_board),
    ]


def _history_planes(hist: dict, me, h: int, w: int, full: bool = False) -> list:
    """The move-history planes of player `me` (see PLANE_NAMES). full: every move seen."""
    states, actions = hist["states"], hist["actions"]
    k_ticks, n = actions.shape[:2]
    weight = (k_ticks - jnp.arange(k_ticks)) / k_ticks                       # newest 1.0 ... oldest 1/K
    rr, cc = jnp.arange(h)[:, None], jnp.arange(w)[None, :]

    def one_tick(st, acts, wk):
        vis = jnp.ones_like(st.mountains) if full else get_visibility(st.ownership[me])
        moving = acts[:, 0] == 0
        r, c = acts[:, 1], acts[:, 2]
        dr, dc = DIRECTIONS[acts[:, 3], 0], DIRECTIONS[acts[:, 3], 1]
        r2, c2 = jnp.clip(r + dr, 0, h - 1), jnp.clip(c + dc, 0, w - 1)
        src = (rr[None] == r[:, None, None]) & (cc[None] == c[:, None, None])        # (n, H, W)
        dst = (rr[None] == r2[:, None, None]) & (cc[None] == c2[:, None, None])
        mine = jnp.arange(n) == me
        seen = moving & ~mine & (vis[r, c] | vis[r2, c2])
        own_from = (src & (moving & mine)[:, None, None]).any(0)
        own_to = (dst & (moving & mine)[:, None, None]).any(0)
        en_from = src & seen[:, None, None]                                      # (n, H, W), by player
        en_to = dst & seen[:, None, None]
        enemy_cells = (st.ownership & ~mine[:, None, None]).any(0) & vis
        en_army = _army_plane(jnp.where(enemy_cells, st.armies, 0))
        return jnp.stack([own_from, own_to], 0) * wk, en_from * wk, en_to * wk, en_army * wk

    own, en_from, en_to, army = (x.max(0) for x in jax.vmap(one_tick)(states, actions, weight))
    by_slot = lambda x: [x[(me + k) % n] if k < n else jnp.zeros((h, w), x.dtype) for k in _OPP_SLOTS]
    return [own[0], own[1], *by_slot(en_from), *by_slot(en_to), army]


def observe(state: GameState, me, hist: dict | None = None, full: bool = False) -> dict:
    """Observation of player `me` (may be a traced int) with fog of war. Free-for-all only (no teams).
    hist: what happened before (history_init / history_push); None means the start of a game.
    full: the same planes without fog, every cell visible and remembered as it is now: the privileged view of a critic
    that is only used in training."""
    if hist is None:
        hist = history_init(state)
    n = state.ownership.shape[0]
    hh, ww = state.armies.shape
    f = lambda x: x.astype(jnp.float32)
    log = lambda x: jnp.log1p(jnp.maximum(f(x), 0.0))

    slots = (me + jnp.arange(n)) % n                       # slot k -> player index
    own_by_slot = state.ownership[slots]                   # (n, H, W), [0] = me
    visible = jnp.ones_like(state.mountains) if full else get_visibility(own_by_slot[0])
    structures = state.mountains | state.castles
    enemy = jnp.any(own_by_slot[1:], axis=0) & visible
    neutral = state.ownership_neutral & visible
    army = jnp.where(visible, state.armies, 0)

    slot_planes = [f(own_by_slot[k] & visible) if k < n else jnp.zeros_like(army, jnp.float32)
                   for k in range(1, MAX_PLAYERS)]
    planes = jnp.stack([
        _army_plane(army) * f(own_by_slot[0]),
        _army_plane(army) * f(enemy),
        _army_plane(army) * f(neutral),
        f(own_by_slot[0]),
        f(neutral),
        f(state.generals & visible),
        f(state.castles & visible),
        f(state.mountains & visible),
        f(~visible & ~structures),
        f(~visible & structures),
        *slot_planes,
        *[f(x) for x in _history_planes(hist, me, hh, ww, full)],
        *_memory_planes(hist, state, me, f, full),
    ], axis=-1)

    # Public scoreboard: land and army of every player (generals.io shows these to everyone).
    land = jnp.sum(own_by_slot, axis=(1, 2))
    total_army = jnp.sum(state.armies[None] * own_by_slot, axis=(1, 2))
    alive = ~state.eliminated[slots]
    t = jnp.full((n,), state.time)
    now = jnp.stack([land, total_army], -1)                                    # (n, 2), by slot
    past = hist["score"][:, slots]                                             # (SCORE_HISTORY, n, 2)
    slog = lambda x: jnp.sign(x) * jnp.log1p(jnp.abs(f(x)))
    changes = [slog(now[:, q] - past[lag - 1, :, q]) for lag in SCORE_LAGS for q in (0, 1)]
    rows = jnp.stack([
        f(jnp.arange(n) == 0), f(alive), log(land), log(total_army),
        f(land) / jnp.maximum(land.sum(), 1), f(total_army) / jnp.maximum(total_army.sum(), 1),
        f(t) / 500.0, f(t % 50) / 50.0, *changes, f(t % 2),
    ], axis=-1)
    players = jnp.zeros((MAX_PLAYERS, rows.shape[1]), jnp.float32).at[:n].set(rows)
    # Players that are out (never present in a padded replay, captured, or left) are masked out entirely.
    player_mask = jnp.zeros((MAX_PLAYERS,), bool).at[:n].set(alive)

    return dict(planes=planes, players=players, player_mask=player_mask, action_mask=action_mask(state, me))
