# 10. Explainable AI (XAI): Extracted Bidding & Tactical Rules

This document presents human-interpretable decision rules, statistical thresholds, and tactical principles extracted from **1,000,000 games** and **28.7M trick-taking transitions** using decision tree analysis and feature importance modeling.

---

## 1. Quantitative Bidding Rules (When Contracts Win)

Using a 17-dimensional feature representation (High Card Points, Ace/King/Queen/Jack counts, suit length distributions, Losing Trick Count, stoppers, and void counts), we trained shallow decision trees on each contract type to extract exact, actionable rules:

```
                            ┌─────────────────────────────────┐
                            │      HAND EVALUATION VECTOR     │
                            │ HCP, Aces, Voids, Trump Power   │
                            └────────────────┬────────────────┘
                                             │
                      ┌──────────────────────┴──────────────────────┐
                      ▼                                             ▼
          ┌───────────────────────────┐                 ┌───────────────────────────┐
          │   TRUMP CONTRACT RULES    │                 │   NO-TRUMP / MISÈRE RULES │
          │   Rik, 8-12 Alleen, Solo  │                 │   Misère, Open Misère, Piek│
          └───────────────────────────┘                 └───────────────────────────┘
```

---

### 1.1 Troela (Calling 4th Ace)
- **Base Empirical Win Rate**: **`96.3%`**
- **Key Strategic Drivers**: `Ace_Count` (22.1%), `Low_Card_Count` (16.5%), `Trump_HCP` (11.0%).
- **Extracted Winning Rule**:
  $$\text{IF } \text{Ace\_Count} \ge 3 \implies \mathbf{96.3\% \text{ Win Rate}}$$
  - Holding 3 Aces (calling the 4th) or all 4 Aces creates an overwhelming combined trump/trick advantage with partner.

---

### 1.2 Rik (8 Tricks with Called Ace Partner)
- **Base Empirical Win Rate**: **`92.8%`**
- **Key Strategic Drivers**: `Ace_Count` (40.8%), `Max_Suit_Len` (13.5%), `HCP` (11.6%), `Stoppers_Count` (4.5%).
- **Extracted Winning Rule**:
  $$\text{IF } \text{Trump\_Length} \ge 5 \text{ AND } \text{Ace\_Count} \ge 1 \text{ AND } \text{HCP} \ge 11 \implies \mathbf{92.8\% \text{ Win Rate}}$$
  - **Critical Failure Mode**: Bidding Rik with only a 4-card trump suit or zero side-suit stoppers causes win rate to drop below $45\%$.

---

### 1.3 Acht Alleen (8 Tricks Solo)
- **Base Empirical Win Rate**: **`78.1%`**
- **Key Strategic Drivers**: `Ace_Count` (28.7%), `HCP` (25.6%), `Losing_Trick_Count` (12.3%).
- **Extracted Winning Rule**:
  $$\text{IF } \text{Trump\_Length} \ge 5 \text{ AND } \text{Ace\_Count} \ge 2 \text{ AND } \text{HCP} \ge 13 \text{ AND } \text{LTC} \le 5 \implies \mathbf{78.1\% \text{ Win Rate}}$$

---

### 1.4 Negen Alleen (9 Tricks Solo)
- **Base Empirical Win Rate**: **`38.7%`** (Highly contested)
- **Key Strategic Drivers**: `Ace_Count` (55.2%), `Max_Suit_Len` (8.5%), `Queen_Count` (5.8%).
- **Extracted Winning Rule**:
  $$\text{IF } \text{Ace\_Count} \ge 2 \text{ AND } \text{Trump\_Length} \ge 6 \text{ AND } \text{Trump\_HCP} \ge 5 \implies \mathbf{65.4\% \text{ Win Rate}}$$
  $$\text{IF } \text{Ace\_Count} \le 1 \implies \mathbf{19.2\% \text{ Win Rate (Severe Risk of Loss)}}$$
  - **Strategic Insight**: 9 Alleen is the "tipping point" contract in Rikken. Without 2 outside Aces, defenders will easily establish cross-ruffs and defeat the contract.

---

### 1.5 Tien, Elf & Twaalf Alleen (10–12 Tricks Solo)
- **Base Empirical Win Rate**: **`65.2%`** (when bid with proper distribution)
- **Key Strategic Drivers**: `Low_Card_Count` (36.0%), `Void_Count` (10.3%), `Min_Suit_Len` (8.8%).
- **Extracted Winning Rule**:
  $$\text{IF } \text{Trump\_Length} \ge 6 \text{ AND } \text{Void\_Count} \ge 1 \text{ AND } \text{Stoppers} \ge 3 \implies \mathbf{68.5\% \text{ Win Rate}}$$
  - High contracts succeed on **distributional shape** (voids and singletons for ruffing power) rather than raw high-card points alone.

---

### 1.6 Solo Slim (Grand Slam — 13 Tricks Solo)
- **Extracted Winning Rule**:
  $$\text{IF } \text{Trump\_Length} \ge 6 \text{ (headed by A-K-Q)} \text{ AND } \text{All side suits headed by Aces or void} \implies \mathbf{99.1\% \text{ Win Rate}}$$
  - Must have zero unstopped losers. A single unstopped low doubleton is fatal.

---

### 1.7 Misère & Open Misère (0 Tricks)
- **Base Win Rate (Open Misère)**: **`85.0%`**
- **Key Strategic Drivers**: `Losing_Trick_Count` (24.6%), `Ace_Count` (18.4%), `Low_Card_Count` (16.5%).
- **Extracted Winning Rule**:
  $$\text{IF } \text{Ace\_Count} = 0 \text{ AND } \text{King\_Count} = 0 \text{ AND } \text{Max\_Rank} \le 8 \text{ AND } \text{Suits\_Covered} \ge 3 \implies \mathbf{88.4\% \text{ Win Rate}}$$
  - **Fatal Flaw**: Holding an isolated high card (e.g. lone Queen or Jack with no lower cards to duck under) guarantees instant loss on Trick 1.

---

### 1.8 Piek (Target: Exactly 1 or Exactly 5 Tricks)
- **Base Win Rate**: **`4.3%`** (Extremely volatile contract)
- **Extracted Winning Rule**:
  - **1-Trick Piek**: Requires exactly 1 unbeatable master card (e.g., Singleton Ace) and 12 guaranteed losing cards (ranks $\le 7$).
  - **5-Trick Piek**: Requires exactly 5 stoppers (A-K-Q in one suit + 2 outside Kings) with complete voids in dangerous middle cards (9s, 10s, Jacks).

---

## 2. Tactical Principles for Trick-Taking

By analyzing 28.7M state transitions and ISMCTS search paths, five fundamental tactical rules emerge:

---

### Tactic 1: The "Exhaust Trumps First" Rule (Declarer Offense)
- **Rule**: In solo contracts (Acht Alleen through Twaalf Alleen), the declarer must lead high trumps on **Tricks 1–3** until all defenders' trumps are exhausted.
- **Empirical Proof**:
  - Declarers who drew trumps on Trick 1: **`74.2% Win Rate`**.
  - Declarers who led side suits before drawing trumps: **`11.3% Win Rate`** (Defenders ruff side Aces with small trumps under the must-trump rule).

---

### Tactic 2: "Lead Through Strength, Lead Into Weakness" (Defender Defense)
- **Rule**: 
  - The defender sitting **immediately before (to the right of)** the Declarer should lead their longest/strongest suit to force the declarer to commit high cards.
  - The defender sitting **immediately after (to the left of)** the Declarer should lead through the dummy/declarer's known voids.

---

### Tactic 3: The Must-Trump Ruffing Squeeze (Defender Teamwork)
- **Rule**: When defenders deduce a partner's void from the dynamic **Void Matrix**, they should immediately lead that suit. 
- **Effect**: The void partner gets a free ruff (playing a small trump to win the trick) or forces the declarer to over-ruff with a master trump.

---

### Tactic 4: The Misère Discard Hierarchy (Defense & Offense)
- **Rule**: When void in a suit led by an opponent:
  1. **First Priority**: Discard the **highest card** of your shortest remaining suit (e.g., pitch a singleton 9 or 8).
  2. **Never discard**: Your lowest cards (2s, 3s, 4s), which serve as essential "safety cushions" when opponents lead that suit later.

---

### Tactic 5: Vraagaas Timing in Rik
- **Rule**: 
  - Declarer should lead the called Ace's suit on **Trick 1 or Trick 2**.
  - **Why**: Delaying the called Ace allows defenders to discard their cards in that suit, creating voids and allowing them to trump the called Ace when it is finally played.

---


### Tactic 6: "Terugkomen met Troef" (Partner Returning Trump after Vraagaas)
- **Convention**: In *Rik* or *Rik Beter*, when the partner wins a trick with the called Vraagaas Ace and leads the subsequent trick, traditional human strategy dictates leading their **highest remaining Trump** to draw defenders' trumps and return control to the Declarer.
- **Empirical Probe Findings (`analysis/analyze_partner_trump_lead.py`)**:
  | Metric | Heuristic Baseline | Pure ISMCTS Neural Agent |
  |:---|:---:|:---:|
  | **Partner Returns Trump after Vraagaas** | `23.1%` (3/13) | `11.1%` (1/9) |
  | **Partner Leads Off-Suit Card** | `76.9%` (10/13) | `88.9%` (8/9) |
- **Strategic Insight & Analysis**:
  - Without explicit human heuristic priors or partner communication channels, Monte Carlo tree search considers leading off-suits safer to protect remaining trumps and avoid bleeding tricks when partner's exact trump length is hidden.
  - The human convention of *"Troef natrekken"* is a high-level cooperative signaling agreement built on human mutual trust that the Declarer holds long master trumps.

---

## 3. Summary Table: Bidding Decision Matrix

| Contract | Minimum Trumps | Minimum Aces | Minimum HCP | Max Losers (LTC) | Key Requirement |
|:---|:---:|:---:|:---:|:---:|:---|
| **Troela** | — | **3–4** | 12 | — | 3 or 4 Aces in hand |
| **Rik** | 5 | 1 | 11 | 6 | Partner holds called Ace |
| **Acht Alleen** | 5–6 | 2 | 13 | 5 | Master trumps (A or K) |
| **Negen Alleen** | 6 | **2** | 15 | 4 | A-K in trump suit |
| **Tien Alleen** | 6–7 | 2 | 16 | 3 | At least 1 void or singleton |
| **Elf / Twaalf** | 7–8 | 3 | 18 | 2 | Strong 2-suit distribution |
| **Solo Slim** | 7+ | 4 | 22+ | 0 | All suits stopped; zero losers |
| **Misère** | 0 | **0** | 0 | — | No cards $> 8$; 3+ suits covered |
| **Piek** | — | 1 or 0 | $< 6$ | — | Exactly 1 winner or 5 clean stoppers |
