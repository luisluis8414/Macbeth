"""Let a trained policy play on generals.io: in the public FFA queue (--ffa) or in a custom game (--lobby).

    uv run scripts/play_online.py --run runs/selfplay --user-id "$GENERALS_USER_ID" --ffa --games 3
    uv run scripts/play_online.py --run runs/selfplay --user-id "$GENERALS_USER_ID" --lobby <name>

The user id is a secret string you choose; whoever knows it can play as that bot. The first time, register a name
for it with --username. With --lobby, open https://generals.io/games/<name> in the browser, join and press Force
Start; the bot joins the lobby, force-starts, plays and joins again after each game. It runs on the CPU. See
src/generals_rl/online.py for how the server's view becomes an observation.

The terminal shows the scoreboard every 25 turns and whenever a player is out, and counts the generals the policy
takes.

While it runs, http://localhost:8052 shows the game live as the bot sees it, with its move probabilities and value.
Every game is logged to runs/online/<replay id>.jsonl: a header, one line per move (the move, its log-probability
under the policy that was played, the value, the five most likely moves) and the result. Together with the replay
(all players' moves) this is enough to rebuild every position for analysis or off-policy training.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from generals.remote import GeneralsIOClient

from macbeth.features import decode_action, history_init, history_push, observe
from macbeth.model import masked_logits
from macbeth.online import LiveView, all_actions, state_from_server
from macbeth.play import load_policy


class PolicyClient(GeneralsIOClient):
    def __init__(self, user_id: str, run: str, ckpt: str, greedy: bool, temperature: float, view: LiveView,
                 log_dir: str):
        apply_fn, params, cfg = load_policy(run, ckpt)
        self.pad = cfg["pad"]
        self.view = view
        self.hist = None
        self.log_dir, self.log = Path(log_dir), None
        self.setup = dict(run=run, ckpt=ckpt, greedy=greedy, temperature=temperature, pad=self.pad)

        def step(state, hist, key):
            """The policy's move (sampled at the given temperature, or the most likely one) and what it thinks of the
            position: value, pass probability, probability per cell, log-probability of the move and the top 5."""
            obs = observe(state, 0, hist)
            logits, value = apply_fn(params, jax.tree.map(lambda a: a[None], obs))
            logits = masked_logits(logits, obs["action_mask"][None])[0]
            if not greedy:
                logits = logits / temperature
            idx = jnp.argmax(logits) if greedy else jrandom.categorical(key, logits)
            action = decode_action(idx, self.pad, self.pad)
            probs = jax.nn.softmax(logits)
            per_cell = probs[:-1].reshape(self.pad, self.pad, -1).sum(-1)
            top_p, top_i = jax.lax.top_k(probs, 5)
            moves = all_actions(action, state.ownership.shape[0])
            return (action, history_push(hist, state, moves), value[0], probs[-1], per_cell, jnp.log(probs[idx]),
                    decode_action(top_i, self.pad, self.pad), top_p)
        self._step = jax.jit(step)
        self.key = jrandom.PRNGKey(int(time.time()))
        super().__init__(agent=None, user_id=user_id, public_server=True)

    def join_private_lobby(self, lobby_id: str) -> None:
        """As in generals-bots, but stop on the server's error instead of waiting silently."""
        self._status = "lobby"
        event, *data = self._emit_receive("join_private", (lobby_id, self.user_id, self.bot_key))
        if event == "gio_error":
            raise SystemExit(f"the server refused to join {lobby_id}: {data[0]}")
        self._queue_id = lobby_id
        print(f"Joined private lobby {lobby_id}.", flush=True)

    def join_ffa_queue(self) -> None:
        """The public FFA queue ("play" in the generals.io bot API); the game starts when the queue fills up."""
        self._status = "queue"
        self.emit("play", (self.user_id, self.bot_key))
        while True:
            event, *data = self.receive()
            if event == "gio_error":
                raise SystemExit(f"the server refused the FFA queue: {data[0]}")
            if event == "game_start":
                self._status = "game"
                self._initialize_game(data)
                self._play_game()
                return

    def _play_game(self) -> None:
        """As in generals-bots, without its 1v1-only Observation (our moves come from the server state itself), and
        without its catch-all that counted any message of unexpected length (chat, game_lost in FFA) as a win."""
        seen = set()
        while True:
            event, *data = self.receive()
            if event == "game_update":
                self.game_state.update(data[0])
                action = self._generate_action(None)
                if action:
                    self.emit("attack", action)
            elif event in ("game_lost", "game_won"):
                self._finish_game(event == "game_won")
                return
            elif event not in seen:                    # anything else (chat, stars, ...) is ignored, logged once
                seen.add(event)
                print(f"server event ignored: {event}", flush=True)

    def _initialize_game(self, data) -> None:
        super()._initialize_game(data)
        self.hist = None
        self.kills, self.out, self.generals_seen = [], set(), {}
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log = open(self.log_dir / f"{self.replay_id}.jsonl", "w")
        gs = self.game_state
        self._write(dict(kind="start", replay=self.replay_id, usernames=gs.usernames, player_index=gs.player_index, time=time.time(), **self.setup))
        self.view.status(f"playing against {', '.join(u for i, u in enumerate(self.game_state.usernames) if i != self.game_state.player_index)}")
        print(f"opponents: {self.game_state.usernames}", flush=True)

    def _write(self, row: dict) -> None:
        if self.log:
            self.log.write(json.dumps(row) + "\n")
            self.log.flush()

    def _finish_game(self, is_winner: bool) -> None:
        turn = getattr(self.game_state, "turn", None)
        self._write(dict(kind="end", won=is_winner, turn=turn, kills=self.kills, time=time.time()))
        if hasattr(self.game_state, "scores"):
            self._print_scoreboard(self.game_state)
        print(f"{'won' if is_winner else 'lost'} at turn {(turn or 0) // 2}, generals taken: {len(self.kills)}"
              + (f" ({', '.join(self.kills)})" if self.kills else ""), flush=True)
        if self.log:
            self.log.close()
            self.log = None
        super()._finish_game(is_winner)
        self.view.status(("won" if is_winner else "lost") + f" · score {self._score_wins}:{self._score_losses}",
                         replay=f"https://generals.io/replays/{self.replay_id}")

    def _generate_action(self, observation):
        gs = self.game_state
        state = state_from_server(gs, self.pad)
        if self.hist is None:
            self.hist = history_init(state)
        self.key, k = jrandom.split(self.key)
        action, self.hist, value, pass_prob, per_cell, logp, top, top_p = self._step(state, self.hist, k)
        w = gs.map[0]

        def server_move(a):
            passed, r, c, d, split = (int(x) for x in a)
            dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[d]
            return None if passed else (r * w + c, (r + dr) * w + c + dc, split)
        move = server_move(action)
        self._follow_scores(gs, float(value))
        self.view.update(gs, move, float(value), float(pass_prob), np.asarray(per_cell))
        me = {s["i"]: s for s in gs.scores}.get(gs.player_index, {})
        self._write(dict(kind="move", turn=gs.turn, move=move, logp=round(float(logp), 4), value=round(float(value), 4),
                         pass_prob=round(float(pass_prob), 4), land=me.get("tiles"), army=me.get("total"),
                         top=[[server_move(a), round(float(q), 4)] for a, q in zip(np.asarray(top), np.asarray(top_p))]))
        return move


    def _follow_scores(self, gs, value: float) -> None:
        """The scoreboard in the log every 25 turns (the land bonus), and a line whenever a player is out: by us
        (their general's cell is now ours) or by someone else. Counts the generals the policy took."""
        me, w, h = gs.player_index, gs.map[0], gs.map[1]
        owner = gs.map[2 + w * h: 2 + 2 * w * h]
        for i, g in enumerate(gs.generals):
            if g >= 0:
                self.generals_seen[i] = g
        event = False
        for s in gs.scores:
            i = s["i"]
            if s["dead"] and i not in self.out:
                self.out.add(i)
                event = True
                if i == me:
                    print(f"turn {gs.turn // 2}: we are out", flush=True)
                elif self.generals_seen.get(i) is not None and owner[self.generals_seen[i]] == me:
                    self.kills.append(gs.usernames[i])
                    print(f"turn {gs.turn // 2}: we took the general of {gs.usernames[i]} "
                          f"(generals taken: {len(self.kills)})", flush=True)
                else:
                    print(f"turn {gs.turn // 2}: {gs.usernames[i]} is out", flush=True)
        if event or gs.turn % 50 == 0:
            self._print_scoreboard(gs, value)

    def _print_scoreboard(self, gs, value: float | None = None) -> None:
        """One line per player, by army: land, army, out or not; ours marked, with the policy's value estimate."""
        rows = sorted(gs.scores, key=lambda s: (s["dead"], -s["total"]))
        head = f"turn {gs.turn // 2:4d}  generals taken {len(self.kills)}" + (f"  value {value:+.2f}" if value is not None else "")
        lines = [f"{'>' if s['i'] == gs.player_index else ' '} {gs.usernames[s['i']][:16]:16s} {s['tiles']:4d} land "
                 f"{s['total']:6d} army" + ("  out" if s["dead"] else "") for s in rows]
        print("\n".join([head, *lines, ""]), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--ckpt", default="latest")
    p.add_argument("--user-id", default=os.environ.get("GENERALS_USER_ID"), help="or the variable GENERALS_USER_ID")
    p.add_argument("--ffa", action="store_true", help="the public FFA queue")
    p.add_argument("--lobby", help="a custom game instead, as in generals.io/games/<name>")
    p.add_argument("--games", type=int, default=1, help="with --ffa: how many games to play before stopping")
    p.add_argument("--username", help="register this name for the user id first (once per id; generals.io refuses renaming)")
    p.add_argument("--greedy", action="store_true", help="always the most likely move instead of sampling")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="sample from softmax(logits / T): 1 = as in training, below 1 = fewer unlikely moves")
    p.add_argument("--log-dir", default="runs/online", help="one .jsonl per game: every move and the result")
    p.add_argument("--view-port", type=int, default=8052, help="live view of the bot's games on this port; 0 = none")
    args = p.parse_args()
    if not args.user_id:
        raise SystemExit("--user-id or GENERALS_USER_ID is needed")
    if args.ffa == bool(args.lobby):
        raise SystemExit("give one of --ffa or --lobby")

    view = LiveView(args.view_port)
    if args.view_port:
        print(f"live view: http://localhost:{args.view_port}", flush=True)
    with PolicyClient(args.user_id, args.run, args.ckpt, args.greedy, args.temperature, view, args.log_dir) as client:
        if args.username:
            client.register_agent(args.username)
        if args.ffa:
            for game in range(args.games):
                print(f"game {game + 1} of {args.games}: waiting in the FFA queue", flush=True)
                view.status(f"game {game + 1} of {args.games}: waiting in the FFA queue")
                client.join_ffa_queue()
            print(f"done: {client._score_wins} won, {client._score_losses} lost", flush=True)
            return
        print(f"open https://generals.io/games/{args.lobby}, join and press Force Start", flush=True)
        view.status(f"in lobby {args.lobby}, waiting for Force Start")
        while True:
            if client.status == "off":
                client.join_private_lobby(args.lobby)
            if client.status == "lobby":
                client.join_game()


if __name__ == "__main__":
    main()
