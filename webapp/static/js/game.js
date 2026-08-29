/**
 * webapp/static/js/game.js — Interactive Client Controller with Full 4-Card Trick Delay & Leader Badges
 */

class RikkenAudio {
    constructor() {
        this.ctx = null;
    }

    init() {
        if (!this.ctx) {
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            this.ctx = new AudioContext();
        }
    }

    playCardSnap() {
        try {
            this.init();
            const osc = this.ctx.createOscillator();
            const gain = this.ctx.createGain();
            osc.type = 'triangle';
            osc.frequency.setValueAtTime(480, this.ctx.currentTime);
            osc.frequency.exponentialRampToValueAtTime(130, this.ctx.currentTime + 0.09);
            gain.gain.setValueAtTime(0.3, this.ctx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.01, this.ctx.currentTime + 0.09);
            osc.connect(gain);
            gain.connect(this.ctx.destination);
            osc.start();
            osc.stop(this.ctx.currentTime + 0.09);
        } catch (e) {}
    }

    playTrickWin() {
        try {
            this.init();
            const now = this.ctx.currentTime;
            [523.25, 659.25, 783.99].forEach((freq, i) => {
                const osc = this.ctx.createOscillator();
                const gain = this.ctx.createGain();
                osc.type = 'sine';
                osc.frequency.setValueAtTime(freq, now + i * 0.07);
                gain.gain.setValueAtTime(0.22, now + i * 0.07);
                gain.gain.exponentialRampToValueAtTime(0.01, now + i * 0.07 + 0.25);
                osc.connect(gain);
                gain.connect(this.ctx.destination);
                osc.start(now + i * 0.07);
                osc.stop(now + i * 0.07 + 0.25);
            });
        } catch (e) {}
    }
}

class RikkenGameApp {
    constructor() {
        this.audio = new RikkenAudio();
        this.state = null;
        this.coachEnabled = false;
        this.selectedOpponent = 'player_1';
        this.selectedTrump = -1;
        this.selectedVraagaas = -1;
        this.draggedCardId = null;
        this.isProcessing = false;
        this.showingTrickResolution = false;

        this.initDOMElements();
        this.bindEvents();
        this.startNewGame();
    }

    initDOMElements() {
        this.statusBadge = document.getElementById('game-status-badge');
        this.contractName = document.getElementById('pill-contract-name');
        this.trumpBadge = document.getElementById('pill-trump-badge');
        this.trumpVal = document.getElementById('pill-trump-val');
        this.partnerBadge = document.getElementById('pill-partner-badge');
        this.partnerVal = document.getElementById('pill-partner-val');
        this.aiDiffSelect = document.getElementById('ai-diff-select');
        this.btnCoachToggle = document.getElementById('btn-coach-toggle');
        this.btnNewGame = document.getElementById('btn-new-game');

        this.cardTable = document.getElementById('card-table');
        this.trickArea = document.getElementById('trick-area');
        this.humanHandContainer = document.getElementById('human-hand');

        this.playerZones = {
            0: document.getElementById('player-zone-0'),
            1: document.getElementById('player-zone-1'),
            2: document.getElementById('player-zone-2'),
            3: document.getElementById('player-zone-3'),
        };
        this.leaderBadges = {
            0: document.getElementById('leader-badge-0'),
            1: document.getElementById('leader-badge-1'),
            2: document.getElementById('leader-badge-2'),
            3: document.getElementById('leader-badge-3'),
        };
        this.trickSlots = {
            0: document.getElementById('trick-slot-0'),
            1: document.getElementById('trick-slot-1'),
            2: document.getElementById('trick-slot-2'),
            3: document.getElementById('trick-slot-3'),
        };
        this.trickNumDisplay = document.getElementById('trick-num-display');
        this.trickLeaderDisplay = document.getElementById('trick-leader-display');
        this.piekPill = document.getElementById('piek-target-pill');
        this.piekVal = document.getElementById('piek-val-display');

        this.handPreviews = {
            1: document.getElementById('hand-preview-1'),
            2: document.getElementById('hand-preview-2'),
            3: document.getElementById('hand-preview-3'),
        };
        this.tricksWonDisplays = {
            0: document.getElementById('tricks-won-0'),
            1: document.getElementById('tricks-won-1'),
            2: document.getElementById('tricks-won-2'),
            3: document.getElementById('tricks-won-3'),
        };
        this.actionBubbles = {
            1: document.getElementById('action-bubble-1'),
            2: document.getElementById('action-bubble-2'),
            3: document.getElementById('action-bubble-3'),
        };

        // Modals
        this.biddingModal = document.getElementById('bidding-modal');
        this.biddingContainer = document.getElementById('bidding-buttons-container');
        this.declModal = document.getElementById('declaration-modal');
        this.declSubtitle = document.getElementById('decl-contract-subtitle');
        this.vraagaasSection = document.getElementById('vraagaas-section');
        this.btnConfirmDecl = document.getElementById('btn-confirm-declaration');

        this.gameoverModal = document.getElementById('gameover-modal');
        this.btnPlayAgain = document.getElementById('btn-play-again');

        // Sidebar
        this.tabBtns = document.querySelectorAll('.tab-btn');
        this.tabContents = document.querySelectorAll('.tab-content');
        this.beliefMatrix = document.getElementById('belief-matrix');
        this.voidMatrixGrid = document.getElementById('void-matrix-grid');
        this.coachEvList = document.getElementById('coach-ev-list');
        this.logStream = document.getElementById('log-stream');
        this.oppSubBtns = document.querySelectorAll('.btn-sub');

        // Toast container
        this.toast = document.createElement('div');
        this.toast.className = 'game-toast';
        document.body.appendChild(this.toast);
    }

    showToast(msg, isError = false) {
        this.toast.textContent = msg;
        this.toast.className = `game-toast show ${isError ? 'error' : ''}`;
        clearTimeout(this.toastTimer);
        this.toastTimer = setTimeout(() => {
            this.toast.className = 'game-toast';
        }, 2500);
    }

    showActionBubble(player, text) {
        if (player === 0 || !this.actionBubbles[player]) return;
        const bubble = this.actionBubbles[player];
        bubble.textContent = text;
        bubble.classList.add('show');
        setTimeout(() => bubble.classList.remove('show'), 2200);
    }

    bindEvents() {
        this.btnNewGame.addEventListener('click', () => this.startNewGame());
        this.btnPlayAgain.addEventListener('click', () => {
            this.gameoverModal.style.display = 'none';
            this.startNewGame();
        });

        this.btnCoachToggle.addEventListener('click', () => {
            this.coachEnabled = !this.coachEnabled;
            this.btnCoachToggle.classList.toggle('active', this.coachEnabled);
            if (this.coachEnabled) this.fetchAICoach();
        });

        // Tabs
        this.tabBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                this.tabBtns.forEach(b => b.classList.remove('active'));
                this.tabContents.forEach(c => c.classList.remove('active'));
                btn.classList.add('active');
                document.getElementById(`tab-${btn.dataset.tab}`).classList.add('active');
                if (btn.dataset.tab === 'xai-beliefs') this.fetchXAI();
                if (btn.dataset.tab === 'ai-coach') this.fetchAICoach();
            });
        });

        // Opponent Sub-tabs
        this.oppSubBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                this.oppSubBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.selectedOpponent = btn.dataset.opp;
                this.fetchXAI();
            });
        });

        // Trump Picker
        document.querySelectorAll('#trump-suit-picker .btn-suit').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('#trump-suit-picker .btn-suit').forEach(b => b.classList.remove('selected'));
                btn.classList.add('selected');
                this.selectedTrump = parseInt(btn.dataset.suit);
                this.validateDeclaration();
            });
        });

        // Vraagaas Picker
        document.querySelectorAll('#vraagaas-suit-picker .btn-suit').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('#vraagaas-suit-picker .btn-suit').forEach(b => b.classList.remove('selected'));
                btn.classList.add('selected');
                this.selectedVraagaas = parseInt(btn.dataset.suit);
                this.validateDeclaration();
            });
        });

        this.btnConfirmDecl.addEventListener('click', () => this.submitDeclaration());

        // Drag & Drop
        [this.trickArea, this.cardTable].forEach(target => {
            target.addEventListener('dragover', (e) => {
                e.preventDefault();
                e.dataTransfer.dropEffect = 'move';
            });
            target.addEventListener('drop', (e) => {
                e.preventDefault();
                if (this.draggedCardId !== null) {
                    this.playCard(this.draggedCardId);
                    this.draggedCardId = null;
                }
            });
        });
    }

    async startNewGame() {
        this.isProcessing = false;
        this.showingTrickResolution = false;
        const diff = this.aiDiffSelect.value;
        const res = await fetch('/api/game/new', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ human_seat: 0, ai_difficulty: diff })
        });
        const data = await res.json();
        if (data.success) {
            this.state = data.state;
            this.updateUI();
            this.showToast("New hand dealt!");
            this.checkAndStepAI();
        }
    }

    async submitBid(bidId) {
        this.biddingModal.style.display = 'none';
        const res = await fetch('/api/game/bid', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ bid_id: bidId })
        });
        const data = await res.json();
        this.state = data.state;
        this.audio.playCardSnap();
        this.updateUI();
        this.checkAndStepAI();
    }

    validateDeclaration() {
        if (this.state && this.state.contract.name === 'RIK') {
            this.btnConfirmDecl.disabled = !(this.selectedTrump >= 0 && this.selectedVraagaas >= 0 && this.selectedTrump !== this.selectedVraagaas);
        } else {
            this.btnConfirmDecl.disabled = !(this.selectedTrump >= 0);
        }
    }

    async submitDeclaration() {
        this.declModal.style.display = 'none';
        const res = await fetch('/api/game/declare', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                trump_suit: this.selectedTrump,
                vraagaas_suit: this.selectedVraagaas
            })
        });
        const data = await res.json();
        this.state = data.state;
        this.updateUI();
        this.checkAndStepAI();
    }

    async playCard(cardId) {
        if (!this.state || this.isProcessing || this.showingTrickResolution) return;
        if (this.state.phase !== 'TRICK_TAKING') {
            this.showToast("Bidding is still in progress.", true);
            return;
        }
        if (!this.state.is_human_turn) {
            this.showToast("Waiting for opponents to play...", true);
            return;
        }
        if (!this.state.legal_plays.includes(cardId)) {
            this.showToast("Illegal card! You must follow suit if possible.", true);
            return;
        }

        this.isProcessing = true;
        this.audio.playCardSnap();

        const res = await fetch('/api/game/play', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ card_id: cardId })
        });
        const data = await res.json();
        if (data.success) {
            const prevState = this.state;
            this.state = data.state;

            // If this human play completed a 4th-card trick:
            if (this.state.last_completed_trick && this.state.last_completed_trick.trick_num > prevState.trick_count) {
                await this.animateTrickResolution(this.state.last_completed_trick);
            } else {
                this.updateUI();
            }

            this.isProcessing = false;
            this.checkAndStepAI();
        } else {
            this.isProcessing = false;
            this.showToast(data.message || "Failed to play card", true);
        }
    }

    async animateTrickResolution(completedTrick) {
        this.showingTrickResolution = true;
        const winner = completedTrick.winner;
        const winnerName = ['You (South)', 'West (AI 1)', 'North (AI 2)', 'East (AI 3)'][winner];

        // 1. Render all 4 completed cards
        this.renderTrickCards(completedTrick.cards);

        // 2. Highlight winning card
        if (this.trickSlots[winner]) {
            this.trickSlots[winner].classList.add('winning-card');
        }
        this.audio.playTrickWin();
        this.showToast(`Trick ${completedTrick.trick_num} won by ${winnerName}!`);

        // 3. PAUSE FOR 1.5 SECONDS so user can inspect all 4 cards
        await new Promise(r => setTimeout(r, 1500));

        // 4. Sweep animation towards winner seat
        this.trickArea.className = `center-trick-area sweep-to-${winner}`;
        await new Promise(r => setTimeout(r, 450));

        // 5. Reset trick area
        this.trickArea.className = 'center-trick-area';
        for (let i = 0; i < 4; i++) {
            this.trickSlots[i].innerHTML = '';
            this.trickSlots[i].classList.remove('winning-card');
        }
        this.showingTrickResolution = false;
        this.updateUI();
    }

    async checkAndStepAI() {
        if (this.isProcessing || !this.state) return;
        if (this.state.phase === 'TERMINAL' || this.state.is_human_turn || this.state.needs_declaration) {
            return;
        }

        this.isProcessing = true;
        while (this.state && !this.state.is_human_turn && this.state.phase !== 'TERMINAL' && !this.state.needs_declaration) {
            // Natural human pacing delay (750ms)
            await new Promise(r => setTimeout(r, 750));

            const res = await fetch('/api/game/step_ai', { method: 'POST' });
            const data = await res.json();
            if (!data.success || !data.event || !data.event.stepped) break;

            const ev = data.event;
            const p = ev.player;

            if (ev.action === 'bid') {
                this.showActionBubble(p, `Bid: ${ev.bid_name}`);
            } else if (ev.action === 'play') {
                this.showActionBubble(p, `Plays ${ev.card.display}`);
                this.audio.playCardSnap();
            }

            this.state = data.state;

            // If trick completed on this AI move:
            if (ev.trick_complete && ev.completed_cards) {
                const compInfo = {
                    cards: ev.completed_cards,
                    winner: ev.winner,
                    trick_num: ev.trick_num
                };
                await this.animateTrickResolution(compInfo);
            } else {
                this.updateUI();
            }
        }
        this.isProcessing = false;
        this.updateUI();
    }

    updateUI() {
        if (!this.state) return;

        // 1. Status & Badges
        const isMyTurn = this.state.is_human_turn;
        this.statusBadge.textContent = isMyTurn ? `YOUR TURN (${this.state.phase})` : `Phase: ${this.state.phase}`;
        this.statusBadge.style.borderColor = isMyTurn ? 'var(--gold)' : 'var(--border-accent)';
        this.statusBadge.style.color = isMyTurn ? 'var(--gold)' : 'var(--blue)';

        const c = this.state.contract;
        this.contractName.textContent = c.name !== 'NO_BID' ? c.name : 'No Bid';

        if (c.is_trump && c.trump_suit >= 0) {
            this.trumpBadge.style.display = 'inline-block';
            this.trumpVal.textContent = ['♣ Clubs', '♦ Diamonds', '♥ Hearts', '♠ Spades'][c.trump_suit];
        } else {
            this.trumpBadge.style.display = 'none';
        }

        if (c.name === 'RIK' && c.vraagaas_suit >= 0) {
            this.partnerBadge.style.display = 'inline-block';
            this.partnerVal.textContent = ['♣ Clubs', '♦ Diamonds', '♥ Hearts', '♠ Spades'][c.vraagaas_suit];
        } else {
            this.partnerBadge.style.display = 'none';
        }

        // Piek indicator
        if (c.name === 'PIEK' || c.name === 'OPEN_PIEK') {
            this.piekPill.style.display = 'flex';
            const tw = this.state.tricks_won[0];
            if (tw === 0) this.piekVal.textContent = 'Aim 1 Trick';
            else if (tw === 1) this.piekVal.textContent = 'Locked at 1 (Duck)';
            else if (tw >= 2 && tw <= 4) this.piekVal.textContent = 'Pivot to 5 Tricks!';
            else if (tw === 5) this.piekVal.textContent = 'Locked at 5 (Duck)';
            else this.piekVal.textContent = 'Busted (>5)';
        } else {
            this.piekPill.style.display = 'none';
        }

        // 2. Leader & Active Turn Indicators
        const leader = this.state.trick_leader;
        const current = this.state.current_player;

        for (let p = 0; p < 4; p++) {
            if (this.leaderBadges[p]) {
                this.leaderBadges[p].style.display = (p === leader && this.state.phase === 'TRICK_TAKING') ? 'inline-block' : 'none';
            }
            const zone = this.playerZones[p];
            if (zone) {
                const avatar = zone.querySelector('.player-avatar');
                if (avatar) {
                    avatar.classList.toggle('active-turn', p === current && this.state.phase !== 'TERMINAL');
                }
            }
            this.tricksWonDisplays[p].textContent = `${this.state.tricks_won[p]} tricks`;
            if (p > 0) {
                const count = this.state.hands_count[p];
                this.handPreviews[p].innerHTML = Array(count).fill('<div class="card-back"></div>').join('');
            }
        }

        // 3. Render Human Hand
        this.renderHumanHand();

        // 4. Render Trick Pile (if not currently holding a completed trick)
        if (!this.showingTrickResolution) {
            this.renderTrickCards(this.state.current_trick);
        }

        // 5. Modals Handling
        if (this.state.phase === 'BIDDING' && this.state.is_human_turn) {
            this.showBiddingModal();
        } else {
            this.biddingModal.style.display = 'none';
        }

        if (this.state.needs_declaration) {
            this.showDeclarationModal();
        } else {
            this.declModal.style.display = 'none';
        }

        if (this.state.phase === 'TERMINAL' && !this.showingTrickResolution) {
            this.showGameOverModal();
        }

        // 6. Logs & XAI
        this.renderLogs();
        this.fetchXAI();
        if (this.coachEnabled) this.fetchAICoach();
    }

    renderHumanHand() {
        this.humanHandContainer.innerHTML = '';
        const cards = this.state.human_hand || [];
        const legalPlays = this.state.legal_plays || [];
        const isTurn = this.state.is_human_turn && this.state.phase === 'TRICK_TAKING' && !this.isProcessing && !this.showingTrickResolution;

        cards.forEach((c, idx) => {
            const cardEl = document.createElement('div');
            const isLegal = legalPlays.includes(c.id);
            const colorClass = c.is_red ? 'red' : 'black';
            const legalClass = isTurn ? (isLegal ? 'legal-play' : 'illegal-play') : '';

            cardEl.className = `card ${colorClass} ${legalClass}`;
            cardEl.style.zIndex = idx + 1;
            cardEl.draggable = isTurn && isLegal;

            cardEl.innerHTML = `
                <div class="card-corner top">
                    <span class="card-rank">${c.rank_char}</span>
                    <span class="card-suit">${c.suit_symbol}</span>
                </div>
                <div class="card-center-suit">${c.suit_symbol}</div>
                <div class="card-corner bottom">
                    <span class="card-rank">${c.rank_char}</span>
                    <span class="card-suit">${c.suit_symbol}</span>
                </div>
            `;

            cardEl.addEventListener('click', (e) => {
                e.stopPropagation();
                this.playCard(c.id);
            });

            cardEl.addEventListener('dragstart', (e) => {
                this.draggedCardId = c.id;
                e.dataTransfer.setData('text/plain', c.id);
                cardEl.classList.add('dragging');
            });

            cardEl.addEventListener('dragend', () => {
                cardEl.classList.remove('dragging');
            });

            this.humanHandContainer.appendChild(cardEl);
        });
    }

    renderTrickCards(trickCards) {
        this.trickNumDisplay.textContent = `Trick ${Math.min(13, this.state.trick_count + 1)} / 13`;
        this.trickLeaderDisplay.textContent = `Lead: ${['South', 'West', 'North', 'East'][this.state.trick_leader]}`;

        for (let p = 0; p < 4; p++) {
            const slot = this.trickSlots[p];
            const c = trickCards ? trickCards[p] : null;
            if (c) {
                const colorClass = c.is_red ? 'red' : 'black';
                slot.innerHTML = `
                    <div class="card ${colorClass}" style="margin: 0; width: 100%; height: 100%; transform: none;">
                        <div class="card-corner top">
                            <span class="card-rank">${c.rank_char}</span>
                            <span class="card-suit">${c.suit_symbol}</span>
                        </div>
                        <div class="card-center-suit" style="font-size: 1.3rem;">${c.suit_symbol}</div>
                        <div class="card-corner bottom">
                            <span class="card-rank">${c.rank_char}</span>
                            <span class="card-suit">${c.suit_symbol}</span>
                        </div>
                    </div>
                `;
            } else {
                slot.innerHTML = '';
            }
        }
    }

    showBiddingModal() {
        this.biddingModal.style.display = 'flex';
        this.biddingContainer.innerHTML = '';
        const legalBids = (this.state.legal_bids || []).map(b => b.id);

        const allContracts = [
            { id: 0, name: 'PAS', target: 'Pass turn' },
            { id: 1, name: 'RIK', target: '8 Tricks (Pick Trump + Partner)' },
            { id: 2, name: 'RIK_BETER', target: '8 Tricks (Hearts Trump + Partner)' },
            { id: 3, name: 'ACHT_ALLEEN', target: '8 Tricks (Solo)' },
            { id: 4, name: 'PIEK', target: 'Exactly 1 or 5 Tricks (Solo)' },
            { id: 5, name: 'NEGEN_ALLEEN', target: '9 Tricks (Solo)' },
            { id: 6, name: 'MISERE', target: '0 Tricks (Ducking Solo)' },
            { id: 7, name: 'TIEN_ALLEEN', target: '10 Tricks (Solo)' },
            { id: 8, name: 'OPEN_PIEK', target: 'Exactly 1 or 5 Tricks (Open Hand)' },
            { id: 9, name: 'OPEN_MISERE', target: '0 Tricks (Open Hand)' },
            { id: 10, name: 'ELF_ALLEEN', target: '11 Tricks (Solo)' },
            { id: 11, name: 'TWAALF_ALLEEN', target: '12 Tricks (Solo)' },
            { id: 12, name: 'SOLO_SLIM', target: '13 Tricks (All Tricks)' },
            { id: 13, name: 'TROELA', target: '8 Tricks (Hold 3 Aces, Call 4th)' },
        ];

        allContracts.forEach(c => {
            const isLegal = legalBids.includes(c.id);
            const btn = document.createElement('button');
            btn.className = `btn-bid ${c.id === 0 ? 'pass-btn' : ''}`;
            btn.disabled = !isLegal;
            btn.innerHTML = `
                <span class="bid-name">${c.name}</span>
                <span class="bid-target">${c.target}</span>
            `;
            if (isLegal) {
                btn.addEventListener('click', () => this.submitBid(c.id));
            }
            this.biddingContainer.appendChild(btn);
        });
    }

    showDeclarationModal() {
        this.declModal.style.display = 'flex';
        this.declSubtitle.textContent = `Contract: ${this.state.contract.name}`;
        this.vraagaasSection.style.display = (this.state.contract.name === 'RIK') ? 'block' : 'none';
        this.selectedTrump = -1;
        this.selectedVraagaas = -1;
        this.validateDeclaration();
    }

    showGameOverModal() {
        this.gameoverModal.style.display = 'flex';
        const rew = this.state.reward || 0;
        const banner = document.getElementById('gameover-banner');
        banner.textContent = rew > 0 ? '🎉 VICTORY!' : '💔 DEFEATED';
        banner.style.color = rew > 0 ? 'var(--gold)' : 'var(--red-suit)';
        document.getElementById('gameover-contract-name').textContent = `Contract: ${this.state.contract.name}`;

        for (let p = 0; p < 4; p++) {
            document.getElementById(`go-tricks-${p}`).textContent = `${this.state.tricks_won[p]} Tricks`;
            const pts = (this.state.rewards && this.state.rewards[p]) ? (this.state.rewards[p] > 0 ? `+${this.state.rewards[p]}` : `${this.state.rewards[p]}`) : '-';
            document.getElementById(`go-points-${p}`).textContent = `${pts} pts`;
        }
    }

    renderLogs() {
        this.logStream.innerHTML = '';
        (this.state.log || []).forEach(msg => {
            const el = document.createElement('div');
            el.className = 'log-entry';
            el.textContent = msg;
            this.logStream.appendChild(el);
        });
        this.logStream.scrollTop = this.logStream.scrollHeight;
    }

    async fetchXAI() {
        try {
            const res = await fetch('/api/game/ai_beliefs');
            const data = await res.json();
            if (!data.available) return;

            const oppData = data.opponents[this.selectedOpponent];
            if (oppData) {
                this.beliefMatrix.innerHTML = '';
                const probs = oppData.card_probabilities;
                const ranks = ['2','3','4','5','6','7','8','9','10','J','Q','K','A'];
                const suits = ['♣','♦','♥','♠'];

                for (let s = 0; s < 4; s++) {
                    for (let r = 0; r < 13; r++) {
                        const cardId = s * 13 + r;
                        const prob = probs[cardId] || 0;
                        const cell = document.createElement('div');
                        cell.className = 'belief-cell';
                        const alpha = Math.min(1.0, prob * 1.5);
                        cell.style.backgroundColor = `rgba(59, 130, 246, ${alpha})`;
                        cell.title = `${ranks[r]}${suits[s]}: ${(prob * 100).toFixed(0)}%`;
                        cell.textContent = prob > 0.05 ? `${(prob * 100).toFixed(0)}` : '';
                        this.beliefMatrix.appendChild(cell);
                    }
                }
            }

            const voids = data.void_matrix;
            if (voids) {
                let html = '<div class="void-header">Player</div><div class="void-header">♣</div><div class="void-header">♦</div><div class="void-header">♥</div><div class="void-header">♠</div>';
                const names = ['You', 'West', 'North', 'East'];
                for (let p = 0; p < 4; p++) {
                    html += `<div class="void-row-label">${names[p]}</div>`;
                    for (let s = 0; s < 4; s++) {
                        const isVoid = voids[p][s] === 1;
                        html += `<div class="void-cell ${isVoid ? 'is-void' : ''}">${isVoid ? 'VOID' : 'OK'}</div>`;
                    }
                }
                this.voidMatrixGrid.innerHTML = html;
            }
        } catch (e) {
            console.warn('XAI fetch error:', e);
        }
    }

    async fetchAICoach() {
        try {
            const res = await fetch('/api/game/ai_advice');
            const data = await res.json();
            if (!data.bids_analysis) return;

            this.coachEvList.innerHTML = '';
            data.bids_analysis.forEach(item => {
                const row = document.createElement('div');
                row.className = `coach-ev-item ${item.recommended ? 'recommended' : ''}`;
                row.innerHTML = `
                    <span class="coach-contract-name">${item.contract_name} ${item.recommended ? '⭐' : ''}</span>
                    <div class="coach-ev-bar-wrap">
                        <span class="coach-ev-val">${(item.ev_score * 100).toFixed(1)}% EV</span>
                    </div>
                `;
                this.coachEvList.appendChild(row);
            });
        } catch (e) {
            console.warn('Coach fetch error:', e);
        }
    }
}

window.addEventListener('DOMContentLoaded', () => {
    window.app = new RikkenGameApp();
});
