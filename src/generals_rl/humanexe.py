"""Games against Human.exe (EklipZ's generals-bot, MIT licence, vendored in third_party/generals-bot).

Human.exe is ~100 000 lines of Python with graph and constraint solvers (OR-Tools, prize-collecting Steiner trees),
so it cannot run inside our JAX simulator. Instead it plays in its own simulator, in a separate process with its own
Python 3.14 (scripts/humanexe_host.py), and our policy plays the other seats from here: every turn the host sends the
full board, we build a GameState from it, the policy observes it through features.observe (fog applied as always)
and sends its moves back. The game rules are the bot's simulator's, the same engine it was developed against.

Setup once with scripts/setup_humanexe.sh: clone the bot into third_party/generals-bot, create its .venv with Python 3.14,
install its requirements, build its extensions, write third_party/run_config.txt.
"""
import json
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from generals.core.game import GameState

from .features import PASS, history_init, history_push

ROOT = Path(__file__).resolve().parents[2]
BOT_DIR = ROOT / "third_party" / "generals-bot"
BOT_PYTHON = BOT_DIR / ".venv" / "bin" / "python"
HOST = ROOT / "scripts" / "humanexe_host.py"
LETTERS = "abcdefgh"
DR, DC = (-1, 1, 0, 0), (0, 0, -1, 1)                    # up, down, left, right as in features.DIRECTIONS


def to_txtmap(state: GameState) -> tuple[str, int, int]:
    """The map of a fresh GameState (cut to the part that is not padding) in the bot's text format, plus its size."""
    mountains = np.asarray(state.mountains)
    rows, cols = np.flatnonzero((~mountains).any(1)), np.flatnonzero((~mountains).any(0))
    h, w = int(rows[-1]) + 1, int(cols[-1]) + 1            # maps from our generator start at the top left
    armies, cities, generals = np.asarray(state.armies), np.asarray(state.castles), np.asarray(state.generals)
    owner = np.where(np.asarray(state.ownership).any(0), np.asarray(state.ownership).argmax(0), -1)
    lines = ["|    " * w]
    for r in range(h):
        cells = []
        for c in range(w):
            if mountains[r, c]:
                cell = "M"
            elif generals[r, c]:
                cell = f"{LETTERS[owner[r, c]]}G{armies[r, c]}"
            elif cities[r, c]:
                cell = (LETTERS[owner[r, c]] if owner[r, c] >= 0 else "") + f"C{armies[r, c]}"
            elif owner[r, c] >= 0:
                cell = f"{LETTERS[owner[r, c]]}{armies[r, c]}"
            else:
                cell = ""
            cells.append(cell.ljust(5))
        lines.append("".join(cells).rstrip())
    return "\n".join(lines) + "\n", h, w


def state_from_host(msg: dict, pad: int, mountains: np.ndarray) -> GameState:
    """The full board the host sent, as our GameState padded to pad x pad with mountains."""
    h, w, n = msg["rows"], msg["cols"], len(msg["dead"])
    owner = np.full((pad, pad), -1)
    owner[:h, :w] = np.asarray(msg["owner"]).reshape(h, w)
    armies = np.zeros((pad, pad), np.int32)
    armies[:h, :w] = np.asarray(msg["army"]).reshape(h, w)
    cities = np.zeros((pad, pad), bool)
    generals = np.zeros((pad, pad), bool)
    positions = np.zeros((n, 2), np.int32)
    for i in msg["cities"]:
        cities[i // w, i % w] = True
    for p, i in enumerate(msg["generals"]):
        if i >= 0:
            generals[i // w, i % w] = True
            positions[p] = (i // w, i % w)
    cities &= ~generals
    ownership = np.stack([owner == p for p in range(n)])
    return GameState(
        armies=jnp.asarray(armies), ownership=jnp.asarray(ownership),
        ownership_neutral=jnp.asarray((owner < 0) & ~mountains), generals=jnp.asarray(generals),
        castles=jnp.asarray(cities), mountains=jnp.asarray(mountains), passable=jnp.asarray(~mountains),
        general_positions=jnp.asarray(positions), teams=jnp.arange(n, dtype=jnp.int32),
        eliminated=jnp.asarray(np.asarray(msg["dead"])), time=jnp.asarray(msg["turn"] - 1, jnp.int32),
        winner=jnp.asarray(-1, jnp.int32), pool_idx=jnp.asarray(0, jnp.int32))


def host_moves_to_actions(moves: list, w: int, n: int) -> jnp.ndarray:
    """(n, 5) actions in our format from the host's [source cell, target cell, half] per player (None = pass)."""
    out = np.tile(np.array(PASS, np.int32), (n, 1))
    for p, mv in enumerate(moves):
        if mv is None:
            continue
        (r, c), (r2, c2) = divmod(mv[0], w), divmod(mv[1], w)
        d = next(k for k in range(4) if (r + DR[k], c + DC[k]) == (r2, c2))
        out[p] = (0, r, c, d, int(mv[2]))
    return jnp.asarray(out)


class HumanExe:
    """A running host process; play() runs one game. Use as a context manager."""

    def __init__(self, log_path: str | None = None):
        if not BOT_PYTHON.exists():
            raise SystemExit(f"Human.exe is not set up: {BOT_PYTHON} is missing (run scripts/setup_humanexe.sh)")
        self.log = open(log_path, "a") if log_path else subprocess.DEVNULL
        self.proc = subprocess.Popen([str(BOT_PYTHON), str(HOST)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.log, text=True, bufsize=1)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.proc.stdin.close()
        self.proc.wait(timeout=30)
        if self.log is not subprocess.DEVNULL:
            self.log.close()

    def _send(self, obj):
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def _recv(self) -> dict:
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("the Human.exe host stopped (see its log)")
        return json.loads(line)

    def start_game(self, start: GameState, outside: list[int], viewer: int = -1, turn_time: float = 0.5,
                   seed: int | None = None) -> GameState:
        """Begin a game on the map of `start` (our generator's or a replay's, top left); `outside` are the seats we
        play, Human.exe plays the rest; seed fixes the bot's random draws (its time budgets still vary). Returns the
        first board as our GameState (padded like `start`)."""
        txt, self.h, self.w = to_txtmap(start)
        self.pad, self.mountains, self.outside = start.armies.shape[-1], np.asarray(start.mountains), list(outside)
        self._send(dict(cmd="start", txtmap=txt, outside=sorted(outside), viewer=viewer, turn_time=turn_time, seed=seed))
        self.msg = self._recv()
        self.n = len(self.msg["dead"])
        return state_from_host(self.msg, self.pad, self.mountains)

    def step(self, moves: dict) -> tuple[GameState, jnp.ndarray]:
        """One turn: moves = {seat: our action [pass, row, col, direction, split] or None} for our seats. Returns the
        next board and everyone's executed moves as (n, 5) actions; self.msg holds done, winner and dead."""
        sent = {}
        for seat in self.outside:
            a = moves.get(seat)
            a = None if a is None else [int(x) for x in np.asarray(a)]
            if a is None or a[0] or self.msg["dead"][seat]:
                sent[str(seat)] = None
            else:
                r, c, d, split = a[1:]
                sent[str(seat)] = [r * self.w + c, (r + DR[d]) * self.w + c + DC[d], bool(split)]
        self._send(dict(moves=sent))
        self.msg = self._recv()
        return state_from_host(self.msg, self.pad, self.mountains), host_moves_to_actions(self.msg["moves"], self.w, self.n)

    def stop_game(self):
        if not self.msg["done"]:
            self._send(dict(stop=True))

    def play(self, start: GameState, outside: dict, max_turns: int = 2000, viewer: int = -1,
             turn_time: float = 0.5) -> dict:
        """One whole game. outside: {seat: act} for the seats we play, where act(state, seat, hist) returns our
        action; Human.exe plays every other seat. viewer: a Human.exe seat whose own debug view (pygame window) shows
        the game in real time, turn_time seconds per turn; -1 = no window, as fast as possible.
        Returns the winner (or None at max_turns), the turn each player went out (None = still in) and the turns."""
        state = self.start_game(start, list(outside), viewer, turn_time)
        hist = history_init(state)
        out_at = [None] * self.n
        while not self.msg["done"] and self.msg["turn"] < max_turns:
            moves = {seat: act(state, seat, hist) for seat, act in outside.items() if not self.msg["dead"][seat]}
            new, actions = self.step(moves)
            hist = history_push(hist, state, actions)
            state = new
            for p, dead in enumerate(self.msg["dead"]):
                if dead and out_at[p] is None:
                    out_at[p] = self.msg["turn"]
        if not self.msg["done"]:                           # cut off at max_turns
            self.stop_game()
            return dict(winner=None, out_at=out_at, turns=self.msg["turn"])
        return dict(winner=self.msg["winner"], out_at=out_at, turns=self.msg["turn"])


def policy_actor(apply_fn, params, pad: int, greedy: bool = False, temperature: float = 1.0, seed: int = 0):
    """act(state, seat, hist) for HumanExe.play from a policy (as play.load_policy returns it)."""
    from .features import decode_action, observe
    from .model import masked_logits

    @jax.jit
    def act(state, seat, hist, key):
        obs = observe(state, seat, hist)
        logits, _ = apply_fn(params, jax.tree.map(lambda x: x[None], obs))
        logits = masked_logits(logits, obs["action_mask"][None])[0]
        idx = jnp.argmax(logits) if greedy else jax.random.categorical(key, logits / temperature)
        return decode_action(idx, pad, pad)
    key = [jax.random.PRNGKey(seed)]

    def call(state, seat, hist):
        key[0], k = jax.random.split(key[0])
        return act(state, jnp.int32(seat), hist, k)
    return call
