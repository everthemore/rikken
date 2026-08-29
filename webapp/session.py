"""
webapp/session.py — Game session manager connecting the Rikken engine to web clients.
"""

from __future__ import annotations
import os
import time
import numpy as np
import torch
from typing import Dict, Any, Optional, List, Tuple

from engine.card import (
    NUM_SUITS, NUM_RANKS, ACE_RANK, HEARTS_SUIT, SUIT_MASKS,
    card_id, suit_of, rank_of, card_to_str, hand_to_str
)
from engine.state import RikkenState, Contract, Phase
from engine.game import RikkenGame
from engine.rules import legal_bids, legal_plays
from agents.heuristic import HeuristicAgent
from agents.neural_agent import NeuralAgent
from networks.bvn import BVN
from networks.bn import BeliefNetwork
import config

RANK_CHARS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
SUIT_SYMBOLS = ['♣', '♦', '♥', '♠']
SUIT_NAMES = ['Clubs', 'Diamonds', 'Hearts', 'Spades']


class GameSession:
    """
    Manages an active game of Rikken between a human player and 3 AI agents.
    """

    def __init__(
        self,
        human_seat: int = 0,
        ai_difficulty: str = "neural_master",
        bvn_path: str = "checkpoints/bvn_best.pt",
        bn_path: str = "checkpoints/bn_best.pt",
        rollouts: int = 50,
        determinizations: int = 10,
        seed: Optional[int] = None,
    ):
        self.human_seat = human_seat
        self.ai_difficulty = ai_difficulty
        self.rollouts = rollouts
        self.determinizations = determinizations
        self.rng = np.random.default_rng(seed)
        self.game = RikkenGame(rng=self.rng)

        # Initialize Agents
        self.agents: List[Any] = []
        for p in range(4):
            if p == human_seat:
                self.agents.append(None)  # Human
            else:
                if ai_difficulty.startswith("neural"):
                    agent = NeuralAgent(
                        seat=p,
                        game=self.game,
                        bvn=bvn_path if os.path.exists(bvn_path) else None,
                        bn=bn_path if os.path.exists(bn_path) else None,
                        n_rollouts=rollouts,
                        n_determinizations=determinizations,
                        device="cpu",
                        rng=self.rng,
                    )
                else:
                    agent = HeuristicAgent(seat=p, rng=self.rng)
                self.agents.append(agent)

        # Game state & logs
        self.state: RikkenState = self.game.reset()
        self.log: List[str] = [f"Game initialized. Human is Player {human_seat} (South)."]
        self.last_action_time = time.time()
        self.completed_tricks: List[Dict[str, Any]] = []
        self.last_completed_trick: Optional[Dict[str, Any]] = None
        self.bidding_history: List[Dict[str, Any]] = []

    def advance_ai_turns(self) -> None:
        """Advance all AI turns until human player must act or game terminates."""
        while self.state.phase != Phase.TERMINAL:
            curr = self.state.current_player
            if curr == self.human_seat:
                break
            res = self.step_ai_single()
            if not res.get("stepped", False):
                break

    def step_ai_single(self) -> Dict[str, Any]:
        """Execute a single AI move and return event metadata for animation."""
        if self.state.phase == Phase.TERMINAL or self.state.current_player == self.human_seat:
            return {"stepped": False, "reason": "Not AI turn or terminal"}

        curr = self.state.current_player
        agent = self.agents[curr]
        event = {"stepped": True, "player": curr, "phase": self.state.phase.name}

        if self.state.phase == Phase.BIDDING:
            bid_action = agent.act(self.state)
            c_name = Contract(bid_action).name if bid_action >= 0 else "PAS"
            self.bidding_history.append({"seat": int(curr), "bid_id": int(bid_action), "bid_name": c_name})
            self.log.append(f"Player {curr} ({self._seat_name(curr)}) bids: {c_name}")
            self.state, reward = self.game.step(self.state, bid_action)
            event["action"] = "bid"
            event["bid_name"] = c_name
            event["bid_id"] = bid_action

            # Check if declaration needed by AI
            if self.state.phase == Phase.TRICK_TAKING and Contract.is_trump_contract(self.state.contract):
                if self.state.declarer == curr:
                    trump = agent.declare_trump(self.state)
                    vraagaas = -1
                    if self.state.contract == Contract.RIK:
                        vraagaas = agent.declare_vraagaas(self.state, trump)
                    self.state = self.game.declare(self.state, trump_suit=trump, vraagaas_suit=vraagaas)
                    t_name = SUIT_NAMES[trump]
                    v_str = f" | Calling Ace of {SUIT_NAMES[vraagaas]}" if vraagaas >= 0 else ""
                    self.log.append(f"Player {curr} declares Trump: {t_name}{v_str}")
                    event["declared_trump"] = trump
                    event["declared_vraagaas"] = vraagaas

        elif self.state.phase == Phase.TRICK_TAKING:
            # Check if human needs to declare trump
            if Contract.is_trump_contract(self.state.contract) and self.state.trump_suit < 0 and self.state.declarer == self.human_seat:
                return {"stepped": False, "reason": "Waiting for human declaration"}

            card = agent.act(self.state)
            self.log.append(f"Player {curr} ({self._seat_name(curr)}) plays {card_to_str(card)}")

            prev_trick_count = self.state.trick_count
            prev_leader = self.state.trick_leader
            trick_cards = list(self.state.current_trick)
            trick_cards[curr] = card

            self.state, reward = self.game.step(self.state, card)

            event["action"] = "play"
            event["card"] = self._format_card(card)
            event["trick_complete"] = False

            if self.state.trick_count > prev_trick_count or self.state.phase == Phase.TERMINAL:
                event["trick_complete"] = True
                event["winner"] = int(self.state.trick_leader)
                event["trick_num"] = int(prev_trick_count + 1)
                event["completed_cards"] = [self._format_card(c) if c >= 0 else None for c in trick_cards]
                completed_info = {
                    "trick_num": prev_trick_count + 1,
                    "leader": prev_leader,
                    "winner": self.state.trick_leader,
                    "cards": trick_cards,
                }
                self.last_completed_trick = completed_info
                self.completed_tricks.append(completed_info)

        return event

    def human_bid(self, contract_val: int) -> Tuple[bool, str]:
        """Process human bid action."""
        if self.state.phase != Phase.BIDDING or self.state.current_player != self.human_seat:
            return False, "Not your turn to bid"

        legal = legal_bids(self.state)
        if contract_val not in legal:
            return False, f"Illegal bid: {Contract(contract_val).name}"

        c_name = Contract(contract_val).name if contract_val >= 0 else "PAS"
        self.bidding_history.append({"seat": int(self.human_seat), "bid_id": int(contract_val), "bid_name": c_name})
        self.log.append(f"You (Player {self.human_seat}) bid: {c_name}")
        self.state, reward = self.game.step(self.state, contract_val)

        # Check if human is declarer and needs to declare trump
        if self.state.phase == Phase.TRICK_TAKING and Contract.is_trump_contract(self.state.contract):
            if self.state.declarer == self.human_seat and self.state.contract == Contract.RIK_BETER:
                # Hearts fixed
                self.state = self.game.declare(self.state, trump_suit=HEARTS_SUIT, vraagaas_suit=-1)

        return True, "Bid placed"

    def human_declare(self, trump_suit: int, vraagaas_suit: int = -1) -> Tuple[bool, str]:
        """Process human trump & vraagaas declaration."""
        if self.state.phase != Phase.TRICK_TAKING or self.state.declarer != self.human_seat:
            return False, "Not authorized to declare trump"

        if not (0 <= trump_suit < 4):
            return False, "Invalid trump suit"

        if self.state.contract == Contract.RIK and not (0 <= vraagaas_suit < 4):
            return False, "Must select a valid Vraagaas suit"

        self.state = self.game.declare(self.state, trump_suit=trump_suit, vraagaas_suit=vraagaas_suit)
        t_name = SUIT_NAMES[trump_suit]
        v_str = f" | Calling Ace of {SUIT_NAMES[vraagaas_suit]}" if vraagaas_suit >= 0 else ""
        self.log.append(f"You declare Trump: {t_name}{v_str}")

        return True, "Declaration confirmed"

    def human_play(self, card: int) -> Tuple[bool, str]:
        """Process human card play."""
        if self.state.phase != Phase.TRICK_TAKING or self.state.current_player != self.human_seat:
            return False, "Not your turn to play"

        legal_mask = legal_plays(self.state)
        if not (0 <= card < 52 and legal_mask[card] == 1):
            return False, f"Illegal card play: {card_to_str(card)}"

        self.log.append(f"You play {card_to_str(card)}")

        prev_trick_count = self.state.trick_count
        prev_leader = self.state.trick_leader
        trick_cards = list(self.state.current_trick)
        trick_cards[self.human_seat] = card

        self.state, reward = self.game.step(self.state, card)

        if self.state.trick_count > prev_trick_count or self.state.phase == Phase.TERMINAL:
            completed_info = {
                "trick_num": prev_trick_count + 1,
                "leader": prev_leader,
                "winner": self.state.trick_leader,
                "cards": trick_cards,
            }
            self.last_completed_trick = completed_info
            self.completed_tricks.append(completed_info)

        return True, "Card played"

    def get_state_payload(self) -> Dict[str, Any]:
        """Serialize full state for web client."""
        human_hand = self.state.hands[self.human_seat]
        hand_card_ids = [int(c) for c in np.where(human_hand)[0]]
        hand_card_ids.sort(key=lambda c: (suit_of(c), rank_of(c)))

        legal_b = legal_bids(self.state) if self.state.phase == Phase.BIDDING and self.state.current_player == self.human_seat else []
        legal_p = [int(c) for c in np.where(legal_plays(self.state))[0]] if self.state.phase == Phase.TRICK_TAKING and self.state.current_player == self.human_seat else []

        needs_declaration = (
            self.state.phase == Phase.TRICK_TAKING
            and Contract.is_trump_contract(self.state.contract)
            and self.state.trump_suit < 0
            and self.state.declarer == self.human_seat
        )

        player_bids = []
        roles = []
        for p in range(4):
            curr_val = int(self.state.bids[p])
            curr_name = Contract(curr_val).name if curr_val > 0 else ("PAS" if curr_val == 0 else "—")
            history = [b["bid_name"] for b in self.bidding_history if b["seat"] == p]
            player_bids.append({
                "seat": p,
                "latest_id": curr_val,
                "latest_name": curr_name,
                "history": history,
                "has_passed": curr_val == 0,
            })

            # Determine role during TRICK_TAKING
            if self.state.phase == Phase.TRICK_TAKING or self.state.phase == Phase.TERMINAL:
                if self.state.declarer_mask[p] or p == self.state.declarer:
                    roles.append("DECLARER")
                elif p == self.state.partner and (self.state.partner_revealed or self.state.contract == Contract.TROELA):
                    roles.append("PARTNER")
                else:
                    roles.append("DEFENDER")
            else:
                roles.append("PASSED" if curr_val == 0 else "BIDDER")
        return {
            "phase": self.state.phase.name,
            "player_bids": player_bids,
            "roles": roles,
            "current_player": int(self.state.current_player),
            "human_seat": int(self.human_seat),
            "is_human_turn": bool(self.state.current_player == self.human_seat and self.state.phase != Phase.TERMINAL),
            "needs_declaration": bool(needs_declaration),
            "contract": {
                "id": int(self.state.contract),
                "name": self.state.contract.name if self.state.contract != Contract.NO_BID else "NO_BID",
                "declarer": int(self.state.declarer),
                "partner": int(self.state.partner),
                "trump_suit": int(self.state.trump_suit),
                "vraagaas_suit": int(self.state.vraagaas_suit),
                "is_trump": bool(Contract.is_trump_contract(self.state.contract)),
            },
            "hands_count": [int(np.sum(self.state.hands[p])) for p in range(4)],
            "human_hand": [self._format_card(c) for c in hand_card_ids],
            "legal_bids": [
                {
                    "id": int(b),
                    "name": Contract(b).name if b >= 0 else "PAS",
                    "target_tricks": Contract.target_tricks(Contract(b)) if b >= 0 else 0,
                }
                for b in legal_b
            ],
            "legal_plays": legal_p,
            "current_trick": [
                self._format_card(c) if c >= 0 else None
                for c in self.state.current_trick
            ],
            "trick_leader": int(self.state.trick_leader),
            "trick_count": int(self.state.trick_count),
            "tricks_won": self.state.tricks_won.tolist(),
            "reward": float(self.state.reward) if self.state.reward is not None else None,
            "rewards": self.state.rewards.tolist() if hasattr(self.state, 'rewards') else None,
            "log": self.log[-15:],
            "completed_tricks": self.completed_tricks,
            "last_completed_trick": {
                "trick_num": int(self.last_completed_trick["trick_num"]),
                "leader": int(self.last_completed_trick["leader"]),
                "winner": int(self.last_completed_trick["winner"]),
                "cards": [self._format_card(c) if c >= 0 else None for c in self.last_completed_trick["cards"]],
            } if self.last_completed_trick is not None else None,
        }

    def get_ai_advice(self) -> Dict[str, Any]:
        """Compute live BVN expected value scores for human hand."""
        if self.state.phase != Phase.BIDDING:
            return {"advice": "AI advice is available during bidding."}

        legal = legal_bids(self.state)
        bvn = None
        for a in self.agents:
            if isinstance(a, NeuralAgent) and a.bvn is not None:
                bvn = a.bvn
                break

        if bvn is None and os.path.exists("checkpoints/bvn_best.pt"):
            bvn = BVN().to("cpu")
            ckpt = torch.load("checkpoints/bvn_best.pt", map_location="cpu", weights_only=False)
            sd = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
            bvn.load_state_dict(sd)
            bvn.eval()

        if bvn is None:
            return {"advice": "Neural BVN model not found."}

        hand = self.state.hands[self.human_seat]
        bids = self.state.bids
        ev_scores = bvn.predict_ev(hand=hand, bids=bids, device="cpu")

        bids_analysis = []
        for b in legal:
            bids_analysis.append({
                "contract_id": int(b),
                "contract_name": Contract(b).name if b >= 0 else "PAS",
                "ev_score": float(ev_scores[b]),
                "recommended": b == int(np.argmax([ev_scores[x] for x in legal])),
            })

        bids_analysis.sort(key=lambda x: x["ev_score"], reverse=True)
        return {"bids_analysis": bids_analysis}

    def get_ai_beliefs(self) -> Dict[str, Any]:
        """Extract live Belief Network card probability distribution for opponents."""
        agent = None
        for a in self.agents:
            if isinstance(a, NeuralAgent) and a.bn is not None:
                agent = a
                break

        if agent is None or agent.bn is None:
            return {"available": False, "message": "Belief Network not loaded."}

        p = self.human_seat
        trick_one_hot = np.zeros(52, dtype=np.int8)
        for c in self.state.current_trick:
            if c >= 0:
                trick_one_hot[c] = 1

        probs = agent.bn.predict(
            own_hand=self.state.hands[p],
            played_cards=self.state.played_cards,
            bid_history=self.state.bids,
            current_trick=trick_one_hot,
            void_matrix=self.state.void_matrix,
            my_seat=p,
            device="cpu",
        )

        opponents = [s for s in range(4) if s != p]
        belief_data = {}
        for idx, opp in enumerate(opponents):
            opp_probs = probs[idx]
            belief_data[f"player_{opp}"] = {
                "seat_name": self._seat_name(opp),
                "card_probabilities": [float(prob) for prob in opp_probs],
            }

        return {
            "available": True,
            "opponents": belief_data,
            "void_matrix": self.state.void_matrix.tolist(),
        }

    def _format_card(self, c: int) -> Dict[str, Any]:
        s = suit_of(c)
        r = rank_of(c)
        return {
            "id": int(c),
            "suit": int(s),
            "rank": int(r),
            "suit_symbol": SUIT_SYMBOLS[s],
            "suit_name": SUIT_NAMES[s],
            "rank_char": RANK_CHARS[r],
            "display": f"{RANK_CHARS[r]}{SUIT_SYMBOLS[s]}",
            "is_red": s in (1, 2),
        }

    def _seat_name(self, seat: int) -> str:
        return ["South (You)", "West (AI 1)", "North (AI 2)", "East (AI 3)"][seat]
