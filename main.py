"""
main.py — CLI entry point for the Rikken AI system.

Commands:
    generate    — Phase 1: generate heuristic self-play dataset
    train-bvn   — Phase 2: train Bidding Value Network
    train-bn    — Phase 2: train Belief Network
    play        — Run a single game (heuristic vs heuristic, with logging)
    benchmark   — Play N games and report win-rate statistics

Example:
    python main.py play --verbose
    python main.py generate --games 1000000 --workers 4
    python main.py train-bvn --resume-latest
    python main.py train-bn --resume-latest
    python main.py benchmark --games 1000
"""

import argparse
import logging
import numpy as np

import config


def cmd_play(args):
    """Run one game of Rikken with verbose logging."""
    from engine.game import RikkenGame
    from engine.state import Contract, Phase
    from engine.card import hand_to_str, card_to_str
    from agents.heuristic import HeuristicAgent

    rng = np.random.default_rng(args.seed)
    game = RikkenGame(rng=rng)
    agents = [HeuristicAgent(seat=p, rng=rng) for p in range(4)]

    state = game.reset()
    print("=" * 60)
    print("NEW GAME")
    print("=" * 60)
    for p in range(4):
        print(f"  Player {p}: {hand_to_str(state.hands[p])}")
    print()

    # --- Bidding ---
    print("--- BIDDING ---")
    while state.phase == Phase.BIDDING:
        p = state.current_player
        action = agents[p].act(state)
        contract_name = Contract(action).name if action >= 0 else "PAS"
        print(f"  Player {p} bids: {contract_name}")
        state, reward = game.step(state, action)
        if reward is not None and state.phase == Phase.TERMINAL:
            print("  All passed → Redeal (reward=0)")
            return

    print(f"\n  Contract: {state.contract.name} by Player {state.declarer}")

    # Declaration
    if Contract.is_trump_contract(state.contract):
        trump = agents[state.declarer].declare_trump(state)
        vraagaas = -1
        if state.contract == Contract.RIK:
            vraagaas = agents[state.declarer].declare_vraagaas(state, trump)
            print(f"  Vraagaas suit: {['C','D','H','S'][vraagaas]}")
        state = game.declare(state, trump_suit=trump, vraagaas_suit=vraagaas)
        print(f"  Trump: {['C','D','H','S'][trump]}")

    # --- Trick-taking ---
    print("\n--- TRICKS ---")
    while state.phase == Phase.TRICK_TAKING:
        p = state.current_player
        action = agents[p].act(state)
        if args.verbose:
            print(f"  Player {p} plays: {card_to_str(action)}")
        state, reward = game.step(state, action)

        if reward is not None:
            break

        # Print trick result when complete
        if state.current_trick[0] == -1 and state.trick_count > 0:
            print(f"  → Trick {state.trick_count} won by Player {state.trick_leader} "
                  f"| Scores: {state.tricks_won.tolist()}")

    print("\n--- RESULT ---")
    print(f"  Tricks: {state.tricks_won.tolist()}")
    print(f"  Contract: {state.contract.name}")
    if state.reward is not None:
        winner = "Declarer" if state.reward > 0 else "Defenders"
        print(f"  Winner: {winner} (reward={state.reward:+.0f})")


def cmd_benchmark(args):
    """Run N games and print statistics."""
    from engine.game import RikkenGame
    from engine.state import Contract, Phase
    from agents.heuristic import HeuristicAgent

    rng = np.random.default_rng(args.seed)
    game = RikkenGame(rng=rng)

    declarer_wins = 0
    total_games = 0
    contract_counts = {}
    redeal_count = 0

    for i in range(args.games):
        agents = [HeuristicAgent(seat=p, rng=rng) for p in range(4)]
        state = game.reset()

        # Bidding
        while state.phase == Phase.BIDDING:
            p = state.current_player
            action = agents[p].act(state)
            state, reward = game.step(state, action)
            if reward is not None and state.phase == Phase.TERMINAL:
                redeal_count += 1
                break

        if state.phase == Phase.TERMINAL:
            continue

        # Declaration
        if Contract.is_trump_contract(state.contract):
            trump = agents[state.declarer].declare_trump(state)
            vraagaas = -1
            if state.contract == Contract.RIK:
                vraagaas = agents[state.declarer].declare_vraagaas(state, trump)
            state = game.declare(state, trump_suit=trump, vraagaas_suit=vraagaas)

        # Play
        while state.phase == Phase.TRICK_TAKING:
            p = state.current_player
            action = agents[p].act(state)
            state, reward = game.step(state, action)
            if reward is not None:
                break

        total_games += 1
        cname = state.contract.name
        contract_counts[cname] = contract_counts.get(cname, 0) + 1
        if state.reward and state.reward > 0:
            declarer_wins += 1

    print(f"\nBenchmark: {total_games} games ({redeal_count} redeals)")
    print(f"  Declarer win rate: {declarer_wins/max(total_games,1):.1%}")
    print("\n  Contract distribution:")
    for c, n in sorted(contract_counts.items(), key=lambda x: -x[1]):
        print(f"    {c:20s}: {n:5d} ({n/total_games:.1%})")


def cmd_generate(args):
    from training.data_gen import generate_dataset
    generate_dataset(
        n_games=args.games,
        n_workers=args.workers,
        data_path=args.path,
        games_per_batch=args.batch_size,
        resume=not args.no_resume,
    )


def cmd_train_bvn(args):
    import torch
    torch.manual_seed(config.TORCH_SEED)
    from training.train_bvn import train
    train(
        epochs=args.epochs,
        lr=args.lr,
        data_path=args.path,
        resume=args.resume,
        resume_latest=args.resume_latest,
    )


def cmd_train_bn(args):
    import torch
    torch.manual_seed(config.TORCH_SEED)
    from training.train_bn import train
    train(
        epochs=args.epochs,
        lr=args.lr,
        data_path=args.path,
        resume=args.resume,
        resume_latest=args.resume_latest,
    )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------



def cmd_tournament(args):
    """Run head-to-head tournament evaluation."""
    from training.tournament import run_tournament
    run_tournament(
        n_games=args.games,
        bvn_path=args.bvn,
        bn_path=args.bn,
        rollouts=args.rollouts,
        determinizations=args.determinizations,
        seed=args.seed,
    )

def cmd_loop(args):
    """Run local iterative self-play RL improvement loop."""
    from training.loop import run_self_play_loop
    run_self_play_loop(
        iterations=args.iterations,
        games_per_iter=args.games_per_iter,
        eval_games=args.eval_games,
        rollouts=args.rollouts,
        determinizations=args.determinizations,
        retrain_epochs=args.retrain_epochs,
        buffer_window=args.buffer_window,
        history_file=args.history_file,
    )

def cmd_docs(args):
    """Start a local HTTP server to view the interactive documentation website."""
    import http.server
    import socketserver
    import webbrowser
    import os

    port = args.port
    docs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'docs')
    os.chdir(docs_dir)

    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        url = f"http://localhost:{port}"
        print(f"\n{'='*60}")
        print(f"  Rikken AI Documentation Website Running")
        print(f"  URL: {url}")
        print(f"  Serving directory: {docs_dir}")
        print(f"  Press Ctrl+C to stop the server.")
        print(f"{'='*60}\n")
        if not args.no_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

def main():
    parser = argparse.ArgumentParser(
        description='Rikken AI — command-line interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--seed', type=int, default=config.NUMPY_SEED)
    sub = parser.add_subparsers(dest='command')

    # play
    p_play = sub.add_parser('play', help='Play one game with heuristic agents')
    p_play.add_argument('--verbose', action='store_true')
    p_play.add_argument('--seed', type=int, default=config.NUMPY_SEED)

    # benchmark
    p_bench = sub.add_parser('benchmark', help='Run N games and print stats')
    p_bench.add_argument('--games',  type=int, default=1000)
    p_bench.add_argument('--seed',   type=int, default=config.NUMPY_SEED)

    # generate
    p_gen = sub.add_parser('generate', help='Generate Phase 1 training data')
    p_gen.add_argument('--games',      type=int, default=config.GAMES_PER_SHARD * config.DATA_SHARDS)
    p_gen.add_argument('--workers',    type=int, default=4)
    p_gen.add_argument('--batch-size', type=int, default=2000)
    p_gen.add_argument('--path',       type=str, default=config.DATA_PATH)
    p_gen.add_argument('--no-resume',  action='store_true', help='Do not skip existing shards')

    # train-bvn
    p_bvn = sub.add_parser('train-bvn', help='Train the Bidding Value Network')
    p_bvn.add_argument('--epochs',        type=int,   default=config.NUM_EPOCHS_BVN)
    p_bvn.add_argument('--lr',            type=float, default=config.LEARNING_RATE)
    p_bvn.add_argument('--path',          type=str,   default=config.DATA_PATH)
    p_bvn.add_argument('--resume',        type=str,   default='')
    p_bvn.add_argument('--resume-latest', action='store_true', help='Resume from bvn_checkpoint.pt')

    # tournament
    p_tour = sub.add_parser('tournament', help='Run head-to-head evaluation tournament')
    p_tour.add_argument('--games',            type=int, default=50)
    p_tour.add_argument('--rollouts',         type=int, default=50)
    p_tour.add_argument('--determinizations', type=int, default=10)
    p_tour.add_argument('--bvn',              type=str, default='checkpoints/bvn_best.pt')
    p_tour.add_argument('--bn',               type=str, default='checkpoints/bn_best.pt')

    # loop
    p_loop = sub.add_parser('loop', help='Run local iterative self-play RL loop')
    p_loop.add_argument('--iterations',       type=int, default=3)
    p_loop.add_argument('--games-per-iter',   type=int, default=100)
    p_loop.add_argument('--eval-games',       type=int, default=50)
    p_loop.add_argument('--rollouts',         type=int, default=50)
    p_loop.add_argument('--determinizations', type=int, default=10)
    p_loop.add_argument('--retrain-epochs',   type=int, default=5)
    p_loop.add_argument('--buffer-window',   type=int, default=config.REPLAY_BUFFER_WINDOW)
    p_loop.add_argument('--history-file',     type=str, default='eval_history.json')

    # docs
    p_docs = sub.add_parser('docs', help='Serve the interactive documentation website')
    p_docs.add_argument('--port', type=int, default=8000, help='Port to serve docs on (default: 8000)')
    p_docs.add_argument('--no-browser', action='store_true', help='Do not automatically open browser')

    # train-bn
    p_bn = sub.add_parser('train-bn', help='Train the Belief Network')
    p_bn.add_argument('--epochs',        type=int,   default=config.NUM_EPOCHS_BN)
    p_bn.add_argument('--lr',            type=float, default=config.LEARNING_RATE)
    p_bn.add_argument('--path',          type=str,   default=config.DATA_PATH)
    p_bn.add_argument('--resume',        type=str,   default='')
    p_bn.add_argument('--resume-latest', action='store_true', help='Resume from bn_checkpoint.pt')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    dispatch = {
        'play':      cmd_play,
        'benchmark': cmd_benchmark,
        'generate':  cmd_generate,
        'train-bvn': cmd_train_bvn,
        'train-bn':  cmd_train_bn,
        'docs':      cmd_docs,
        'tournament': cmd_tournament,
        'loop':       cmd_loop,
    }

    if args.command in dispatch:
        dispatch[args.command](args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
