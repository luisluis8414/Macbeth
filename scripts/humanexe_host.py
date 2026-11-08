"""Runs games of Human.exe (EklipZ's generals-bot, third_party/generals-bot) in its own game simulator, with some seats
played from outside. Started by src/generals_rl/humanexe.py with the bot's own Python (third_party/generals-bot/.venv,
Python 3.14); it is not meant to be run by hand.

Protocol, one JSON object per line on stdin/stdout:
    in:  {"cmd": "start", "txtmap": <map in the bot's text format>, "outside": [seats played from outside],
          "viewer": seat of a Human.exe player whose debug view opens in a window (-1 = none), "turn_time": seconds,
          "seed": seed for Python's random module (optional)}
    out: {"turn", "rows", "cols", "owner", "army", "cities", "generals", "dead", "moves", "done", "winner"}
         owner/army: per cell, row by row (owner -1 = nobody); cities: cell indices; generals: cell per player (-1 if
         unknown or gone); moves: each player's move of the last turn as [source cell, target cell, half] or null
    in:  {"moves": {"<seat>": [source cell, target cell, half] or null}}       for every outside seat, every turn
         or {"stop": true} to abandon the game
    ... until "done" is true; then the next "start" or end of input.
Everything the bot prints goes to stderr, so stdout carries only the protocol.
"""
import json
import os
import random
import sys

BOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party", "generals-bot")
protocol = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)                                              # the bot's own output (also from C code) goes to stderr,
sys.stdout = sys.stderr                                    # stdout carries only the protocol
os.chdir(BOT)
sys.path.insert(0, BOT)

import logbook  # noqa: E402

logbook.NullHandler().push_application()

import BotHost  # noqa: E402
from BotModules.BotRendering import BotRendering  # noqa: E402
from base.client.map import MapBase, Score  # noqa: E402
from Models import Move  # noqa: E402
from Sim.GameSimulator import GameSimulatorHost  # noqa: E402
from Sim.TextMapLoader import TextMapLoader  # noqa: E402


def send(obj):
    protocol.write(json.dumps(obj) + "\n")
    protocol.flush()


def snapshot(host, done, winner):
    m = host.sim.sim_map
    tiles = [t for row in m.grid for t in row]
    last = host.sim.moves_history[-1] if host.sim.moves_history else [None] * len(m.players)
    move = lambda mv: None if mv is None else [mv.source.tile_index, mv.dest.tile_index, bool(mv.move_half)]
    return dict(turn=m.turn, rows=m.rows, cols=m.cols, owner=[t.player for t in tiles], army=[t.army for t in tiles],
                cities=[t.tile_index for t in tiles if t.isCity],
                generals=[p.general.tile_index if p.general is not None and not sp.dead else -1
                          for p, sp in zip(m.players, host.sim.players)],
                dead=[bool(sp.dead) for sp in host.sim.players], moves=[move(mv) for mv in last], done=done, winner=winner)


def play(txtmap: str, outside: list[int], viewer: int = -1, turn_time: float = 0.5):
    tiles = TextMapLoader.load_map_from_string(txtmap)
    players = 1 + max(t.player for row in tiles for t in row)
    for row in tiles:
        for t in row:
            t.visible = t.discovered = True
            t.lastSeen = 1
    m = MapBase(player_index=0, teams=None, user_names=[f"p{i}" for i in range(players)], turn=1, map_grid_y_x=tiles,
                replay_url="generals-rl")
    m.update_scores([Score(n, 1, 1, False) for n in range(players)])
    m.update_turn(1)
    m.update(bypassDeltas=True)
    BotHost.FORCE_NO_VIEWER = viewer < 0                   # the bot's repo hard-codes True (no windows at all)
    host = GameSimulatorHost(m, player_with_viewer=viewer, afkPlayers=list(outside))
    for bot in host.bot_hosts:
        if bot is None:
            continue
        bot.rethrow = False                                # a bug in one move (it has some) costs that move, not the game
        if viewer >= 0 and bot.has_viewer:                 # as GameSimulatorHost.run_sim starts its window
            bot.initialize_viewer(bot.eklipz_bot.no_file_logging, onClick=host.on_click)
            bot.run_viewer_loop()
            BotRendering.prep_view_info_for_render(bot.eklipz_bot, None)
            bot._viewer.send_update_to_viewer(bot.eklipz_bot.viewInfo, bot.eklipz_bot._map)
    send(snapshot(host, False, None))
    while True:
        reply = json.loads(sys.stdin.readline())
        if reply.get("stop"):                              # the caller cut the game off
            finish(host, viewer)
            return
        sim_map = host.sim.sim_map
        for seat, mv in reply["moves"].items():
            seat = int(seat)
            if mv is None or host.sim.players[seat].dead:
                host.queue_player_move(seat, None)
                continue
            src, dst = sim_map.tiles_by_index[mv[0]], sim_map.tiles_by_index[mv[1]]
            host.queue_player_move(seat, Move(src, dst, bool(mv[2])))
        winner = host.execute_turn(run_real_time=viewer >= 0, turn_time=turn_time)
        done = winner is not None or host.sim.is_game_over()
        send(snapshot(host, done, winner))
        if done:
            finish(host, viewer)
            return


def finish(host, viewer: int):
    """End the game for the bots; with a window, leave the last board up for a moment."""
    if viewer >= 0:
        host.wait_until_viewer_closed_or_time_elapses(3.0)
    host.sim.end_game()
    for bot in host.bot_hosts:
        if bot is not None:
            bot.notify_game_over()


def main():
    for line in sys.stdin:
        msg = json.loads(line)
        if msg.get("cmd") == "start":
            if msg.get("seed") is not None:                # the bot draws random numbers (timings, ties, openings)
                random.seed(msg["seed"])
            play(msg["txtmap"], msg.get("outside", []), msg.get("viewer", -1), msg.get("turn_time", 0.5))


if __name__ == "__main__":
    main()
