# We Made What - Word-Mixing Game Architecture Plan

## Context

Build a web game where players combine and subtract words via embedding vector arithmetic (similar to InfiniteCraft). E.g., "fire" + "water" = "steam". Players start with seed words, discover new words through combinations, and race to find target words. Supports solo play and 1v1 competitive matches over WebSockets. Vocabulary includes common English words AND pop culture (anime, films, TV, music, actors, video games, memes).

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Backend | **Python 3.13 + uv + FastAPI** | User preference, async WebSocket support built-in |
| Embedding | **Qwen/Qwen3-Embedding-8B** via `sentence-transformers` | Dedicated embedding model, #1 MTEB multilingual, 4096d (Matryoshka: can use 512-1024d for speed), supports instruction-aware embedding. ~16GB VRAM FP16, ~8GB Q8 |
| Vector search | **FAISS `IndexFlatIP`** (CPU, brute-force) | At 512d with ~150k vocab, index is ~300MB, search <2ms. No approximate search needed |
| Vocabulary | **NLTK + wordfreq + Wikidata SPARQL** | ~100-150k entries: common English + pop culture entities |
| Frontend | **React 19 + TypeScript + Vite 6** | Best ecosystem for DnD libraries and real-time game state |
| Drag-and-drop | **@dnd-kit/core** | Hook-based, actively maintained, flexible |
| State mgmt | **Zustand 5** | Minimal boilerplate, great for real-time game state |
| Animations | **Framer Motion 12** | Merge effects, discovery celebrations, fluid game feel |
| Styling | **Tailwind CSS 4** | Rapid styling |

### Embedding Model Details: Qwen3-Embedding-8B

- **HuggingFace**: `Qwen/Qwen3-Embedding-8B`
- **Type**: Dedicated text embedding model (NOT an LLM), Apache 2.0
- **Dimensions**: 4096 native, but supports Matryoshka (32-4096). We'll use **512d** — sweet spot for word arithmetic speed vs quality
- **Inference**: `sentence-transformers` library directly (`model.encode()`)
- **Instruction-aware**: Supports task-specific prompts for better embeddings
- **GPU**: ~16GB FP16, ~8GB Q8. Flash Attention 2 recommended
- **Swappable**: Change `WMW_EMBEDDING_MODEL` env var → rebuild vocab index → restart

## Project Structure

```
we-made-what/
├── backend/
│   ├── pyproject.toml
│   ├── src/wmw/
│   │   ├── main.py              # FastAPI app, lifespan (loads model + index once)
│   │   ├── config.py            # Pydantic Settings (WMW_ env prefix)
│   │   ├── embedding/
│   │   │   ├── provider.py      # EmbeddingProvider Protocol (swap models here)
│   │   │   ├── sentence_tf.py   # SentenceTransformerProvider implementation
│   │   │   └── registry.py      # Model name -> provider factory
│   │   ├── vocab/
│   │   │   ├── index.py         # VocabIndex: FAISS wrapper, nearest-neighbor search
│   │   │   ├── builder.py       # Build/rebuild index from word list + pop culture
│   │   │   ├── wordlist.py      # Load + filter NLTK words via wordfreq
│   │   │   └── popculture.py    # Fetch pop culture entities (Wikidata SPARQL)
│   │   ├── engine/
│   │   │   └── mixer.py         # WordMixer: add/subtract vectors, find nearest word
│   │   ├── game/
│   │   │   ├── session.py       # GameSession state machine (LOBBY->COUNTDOWN->PLAYING->FINISHED)
│   │   │   ├── manager.py       # SessionManager: create/join/leave rooms
│   │   │   ├── matchmaker.py    # Matchmaking queue for 1v1
│   │   │   └── scoring.py       # Scoring rules, win conditions
│   │   ├── api/
│   │   │   ├── routes.py        # REST: health, lobby info
│   │   │   ├── ws.py            # WebSocket endpoint + message dispatcher
│   │   │   └── schemas.py       # Pydantic request/response models
│   │   └── data/                # Vocab index cache + raw word lists (runtime, gitignored)
│   └── tests/
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   └── src/
│       ├── App.tsx
│       ├── stores/
│       │   ├── gameStore.ts     # Game state (words, score, phase, targets)
│       │   └── connectionStore.ts
│       ├── hooks/
│       │   ├── useWebSocket.ts  # Connection + reconnect + message dispatch
│       │   ├── useGameActions.ts
│       │   └── useDragMerge.ts  # DnD-kit combine logic
│       ├── components/
│       │   ├── WordTile.tsx     # Draggable word chip (both draggable + drop target)
│       │   ├── Workspace.tsx    # Main play area - free-form canvas
│       │   ├── WordInventory.tsx # Scrollable discovered words panel
│       │   ├── TargetPanel.tsx  # Target words to find (with progress)
│       │   ├── ScoreBoard.tsx
│       │   ├── SubtractZone.tsx # Drag two words into A/B slots to subtract
│       │   ├── GameOverlay.tsx  # Countdown, results, celebrations
│       │   └── Lobby.tsx
│       └── lib/
│           └── protocol.ts     # WS message type definitions (mirrors backend schemas)
└── scripts/
    ├── build-vocab.py           # One-shot: fetch all words, embed, build FAISS index
    ├── fetch-popculture.py      # Fetch pop culture entities from APIs → raw word lists
    └── evaluate-model.py        # Compare models on word-arithmetic quality
```

## Vocabulary Strategy (~100-150k entries)

### Sources

| Source | What | How | Estimated Count |
|---|---|---|---|
| **NLTK + wordfreq** | Common English words, filtered by frequency | `wordfreq.zipf_frequency()` > threshold | ~40-50k |
| **Wikidata SPARQL** | Film titles, TV shows, actors, musicians, video games, fictional characters | SPARQL queries against `query.wikidata.org` | ~30-40k |
| **Wikidata SPARQL** (anime) | Anime titles, manga titles | SPARQL queries | ~10-15k |

### Build Pipeline (`scripts/fetch-popculture.py` + `scripts/build-vocab.py`)

1. **Fetch** raw word lists from each source → save as JSON in `backend/src/wmw/data/raw/`
2. **Deduplicate** and normalize (lowercase, strip punctuation, keep multi-word phrases as-is like "darth vader")
3. **Tag** each entry with its category (english, anime, film, tv, music, actor, game, character)
4. **Embed** all entries in batches using the configured model (batch_size=256, ~15 min for 150k on GPU)
5. **Build** FAISS `IndexFlatIP` index with L2-normalized vectors
6. **Save** index + word metadata to `backend/src/wmw/data/`

Multi-word entries work fine — embedding models handle phrases like "darth vader" or "studio ghibli" natively.

## Key Design Decisions

### Embedding Dimensions: 512d (via Matryoshka)
Qwen3-Embedding-8B outputs 4096d natively but supports Matryoshka truncation. Using 512d:
- Cuts FAISS index from ~2.3GB to ~300MB
- Faster nearest-neighbor search
- Word arithmetic quality at 512d is still excellent for this use case
- Configurable via `WMW_EMBEDDING_DIM` if we want to experiment

### Embedding Model Swap = Config + Rebuild + Restart
Different models produce incompatible embedding spaces, so the FAISS index must be rebuilt. Flow: change `WMW_EMBEDDING_MODEL` env var, run `build-vocab.py`, restart server.

### Server-Authoritative Game State
All combination logic runs server-side. Client never sees embedding vectors. Anti-cheat by design. Rate limit ~2 combines/sec per player.

### Simultaneous Real-Time Play (Not Turn-Based)
Both players play at the same time in versus mode. More exciting "discovery race" gameplay.

### Exclude Input Words from Results
When computing `fire + water`, exclude "fire" and "water" from nearest-neighbor results to force genuinely new discoveries.

### Pop Culture Categories Visible in UI
When a player discovers "goku", the result tile shows a small tag like `[anime]`. Adds delight and context.

## Data Flow: Word Combination

```
Player drags "fire" onto "water"
  -> Frontend sends WS: { type: "combine", word_a: "fire", word_b: "water" }
  -> Backend validates words in player inventory
  -> WordMixer: vec_fire + vec_water -> normalize -> FAISS nearest (exclude inputs)
  -> Returns "steam" (score 0.82, category: "english")
  -> GameSession: check if new, check if target, update score
  -> WS response: { type: "mix_result", result: "steam", is_new: true, is_target: true, category: "english" }
  -> Frontend: merge animation, add to inventory, update score
  -> If versus: opponent gets throttled score update (1/sec)
```

Total round-trip: ~10-30ms localhost (slightly slower than MiniLM due to larger model, but still instant-feeling).

## WebSocket Protocol

**Client -> Server:** `create_game`, `join_game`, `find_match`, `combine`, `subtract`, `ping`
**Server -> Client:** `connected`, `game_starting`, `game_started`, `mix_result`, `opponent_update`, `game_over`, `error`, `pong`

All JSON with `type` discriminator field. No socket.io needed.

## Game Session Lifecycle

1. **LOBBY** - Player creates game (solo) or enters matchmaking (versus)
2. **COUNTDOWN** - 3s countdown, server sends seed words (5) + target words (10) + time limit (120s)
3. **PLAYING** - Players combine/subtract freely, server processes and responds
4. **FINISHED** - Time up or all targets found. Results comparison screen with all discoveries

**Scoring:** 10pts per target word, 1pt per non-target discovery, bonus for first-to-find in versus.

## UX: Making it Feel Fluid and Fun

- **Drag-and-drop combining**: Drag a word tile onto another → tiles animate merging → result pops in with a satisfying bounce
- **Subtraction zone**: Dedicated area at bottom with "A" and "B" slots. Drag words into slots, hit subtract button. Result animates in. Can also right-click a tile → "subtract from..." mode
- **Discovery celebrations**: New words pop in with spring animation. Target words get a special glow/confetti burst. Pop culture discoveries show their category tag with a themed color
- **Workspace**: Free-form canvas where word tiles can be positioned anywhere (not a grid). Tiles float with subtle idle animations
- **Sound effects** (optional, toggle): Satisfying click/merge/discovery sounds
- **Opponent presence**: In versus mode, a subtle side panel shows opponent's score ticking up and their discovery count — creates tension without revealing their strategy
- **Timer**: Dramatic visual countdown in final 10 seconds
- **Results screen**: Side-by-side comparison of both players' discovery trees

## Implementation Phases

### Phase 1: Core Engine (no web, no game)
1. `pyproject.toml` with uv, install deps (sentence-transformers, faiss-cpu, nltk, wordfreq, httpx)
2. `EmbeddingProvider` protocol + `SentenceTransformerProvider` (with Matryoshka dim support)
3. `wordlist.py` — load NLTK words, filter by `wordfreq`
4. `popculture.py` — fetch from Wikidata SPARQL (all pop culture categories)
5. `scripts/fetch-popculture.py` — run fetchers, save raw lists
6. `VocabIndex` with FAISS (`IndexFlatIP`, L2-normalized vectors, cosine via dot product)
7. `scripts/build-vocab.py` — combine all word sources, embed, build index
8. `WordMixer` — combine/subtract with exclude-inputs logic
9. Tests: verify word arithmetic produces reasonable results

### Phase 2: HTTP API (testable backend)
10. FastAPI app with lifespan (load model + index once at startup)
11. `POST /api/mix` endpoint for testing combinations
12. Health check, vocab stats endpoint

### Phase 3: Single-Player Game
13. `GameSession` state machine
14. `SessionManager`
15. WebSocket endpoint + message dispatcher
16. Seed/target word generation (mix of English + pop culture targets)
17. Timer logic (asyncio.Task)

### Phase 4: Frontend
18. Vite + React + TS scaffold with Tailwind 4
19. Zustand stores (gameStore, connectionStore)
20. WebSocket hook with reconnect + exponential backoff
21. WordTile + Workspace (drag-and-drop via @dnd-kit, free-form positioning)
22. SubtractZone (two-slot drag interface)
23. Inventory panel, target panel with progress, scoreboard
24. Animations: merge (Framer Motion spring), discovery pop-in, countdown, confetti for targets
25. Lobby screen (solo start, matchmaking button)
26. Category tags on discovered words (color-coded: anime=pink, film=gold, etc.)

### Phase 5: Multiplayer
27. Matchmaking queue
28. Opponent state broadcast (throttled 1/sec)
29. Versus scoring (first-to-find bonus)
30. Results comparison screen (side-by-side discovery trees)

## Verification

1. **Engine**: `uv run python -c "from wmw.engine.mixer import WordMixer; ..."` — verify "fire" + "water" gives something reasonable, "naruto" + "dragon" gives something fun
2. **API**: `curl localhost:8000/api/mix -d '{"word_a":"fire","word_b":"water","op":"add"}'`
3. **WebSocket**: Connect via browser devtools, send combine messages, verify responses
4. **Frontend**: Open browser, drag words onto each other, verify merge animation and new word appears in inventory
5. **Tests**: `uv run pytest` for backend
6. **Multiplayer**: Open two browser tabs, matchmake, verify both see game state updates
