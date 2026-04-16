# We Made What - Word-Mixing Game Architecture Plan

## Context

Build a web game where players combine and subtract words creatively (similar to InfiniteCraft). E.g., "fire" + "water" = "firestorm", "sword" + "japan" = "katana". A local LLM generates the combinations. Players start with seed words, discover new words, and race to find target words. Supports solo play and 1v1 competitive matches over WebSockets.

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Backend | **Python 3.13 + uv + FastAPI** | User preference, async WebSocket support built-in |
| LLM | **Gemma-4-E4B-Uncensored** via `llama-cpp-python` | 4B params, ~5GB VRAM Q4_K_M, fast inference, uncensored for creative output |
| Frontend | **React 19 + TypeScript + Vite 6** | Best ecosystem for DnD libraries and real-time game state |
| Drag-and-drop | **@dnd-kit/core** | Hook-based, actively maintained, flexible |
| State mgmt | **Zustand 5** | Minimal boilerplate, great for real-time game state |
| Animations | **Framer Motion 12** | Merge effects, discovery celebrations, fluid game feel |
| Styling | **Tailwind CSS 4** | Rapid styling |

### LLM Details: Gemma-4-E4B-Uncensored

- **HuggingFace**: `HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive`
- **Format**: GGUF (Q4_K_M recommended, ~5GB)
- **Inference**: `llama-cpp-python` with GPU offloading
- **VRAM**: ~5GB on RTX 4070 Ti
- **Swappable**: Change `WMW_MODEL_PATH` env var, restart server

## Project Structure

```
we-made-what/
├── backend/
│   ├── pyproject.toml
│   ├── src/wmw/
│   │   ├── main.py              # FastAPI app, lifespan (loads LLM once)
│   │   ├── config.py            # Pydantic Settings (WMW_ env prefix)
│   │   ├── llm/
│   │   │   ├── provider.py      # LLMProvider Protocol (swap models here)
│   │   │   └── llamacpp.py      # LlamaCppProvider (GGUF via llama-cpp-python)
│   │   ├── engine/
│   │   │   └── mixer.py         # WordMixer: prompt LLM to combine/subtract words
│   │   ├── game/
│   │   │   ├── session.py       # GameSession state machine
│   │   │   ├── manager.py       # SessionManager: create/join/leave rooms
│   │   │   ├── matchmaker.py    # Matchmaking queue for 1v1
│   │   │   └── scoring.py       # Scoring rules, win conditions
│   │   └── api/
│   │       ├── routes.py        # REST: health, lobby info
│   │       ├── ws.py            # WebSocket endpoint + message dispatcher
│   │       └── schemas.py       # Pydantic request/response models
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
│       │   ├── WordInventory.tsx
│       │   ├── TargetPanel.tsx  # Target words to find (with progress)
│       │   ├── ScoreBoard.tsx
│       │   ├── SubtractZone.tsx # Drag two words into A/B slots to subtract
│       │   ├── GameOverlay.tsx  # Countdown, results, celebrations
│       │   └── Lobby.tsx
│       └── lib/
│           └── protocol.ts     # WS message type definitions (mirrors backend schemas)
└── models/                      # GGUF model files (gitignored)
```

## Key Design Decisions

### LLM-Based Word Mixing (Not Embedding Arithmetic)
Instead of vector math (add/subtract embeddings, find nearest neighbor), we prompt a local LLM to creatively generate word combinations. This produces much more fun and varied results ("sword + japan = katana", "cat + internet = catnet") compared to embedding arithmetic which was unreliable.

### Result Caching
Same word combination always returns the same result. Cache is keyed on `(word_a, word_b, operation)` and is case-insensitive. This ensures deterministic gameplay and avoids redundant LLM calls.

### Server-Authoritative Game State
All combination logic runs server-side. Client never calls the LLM directly. Anti-cheat by design. Rate limit ~2 combines/sec per player.

### Simultaneous Real-Time Play (Not Turn-Based)
Both players play at the same time in versus mode. More exciting "discovery race" gameplay.

## Data Flow: Word Combination

```
Player drags "fire" onto "water"
  -> Frontend sends WS: { type: "combine", word_a: "fire", word_b: "water" }
  -> Backend validates words in player inventory
  -> WordMixer: check cache → miss → prompt LLM
  -> LLM generates "firestorm"
  -> Cache result, return to GameSession
  -> GameSession: check if new, check if target, update score
  -> WS response: { type: "mix_result", result: "firestorm", is_new: true }
  -> Frontend: merge animation, add to inventory, update score
  -> If versus: opponent gets throttled score update (1/sec)
```

LLM inference: ~200-500ms per combination on GPU. Cached results: instant.

## WebSocket Protocol

**Client -> Server:** `create_game`, `join_game`, `find_match`, `combine`, `subtract`, `ping`
**Server -> Client:** `connected`, `game_starting`, `game_started`, `mix_result`, `opponent_update`, `game_over`, `error`, `pong`

All JSON with `type` discriminator field. No socket.io needed.

## Game Session Lifecycle

1. **LOBBY** - Player creates game (solo) or enters matchmaking (versus)
2. **COUNTDOWN** - 3s countdown, server sends seed words (5) + target words (10) + time limit (120s)
3. **PLAYING** - Players combine/subtract freely, server processes and responds
4. **FINISHED** - Time up or all targets found. Results comparison screen

**Scoring:** 10pts per target word, 1pt per non-target discovery, bonus for first-to-find in versus.

**Seed words:** Curated static list of good starter words (fire, water, earth, air, metal, etc.)

**Target words:** Pre-generated by running seed word combinations through the LLM offline.

## UX: Making it Feel Fluid and Fun

- **Drag-and-drop combining**: Drag a word tile onto another → tiles animate merging → result pops in with a satisfying bounce
- **Subtraction zone**: Dedicated area with "A" and "B" slots. Drag words into slots, hit subtract button
- **Discovery celebrations**: New words pop in with spring animation. Target words get a glow/confetti burst
- **Workspace**: Free-form canvas where word tiles can be positioned anywhere
- **Opponent presence**: In versus mode, side panel shows opponent's score and discovery count
- **Timer**: Dramatic visual countdown in final 10 seconds
- **Results screen**: Side-by-side comparison of both players' discoveries

## Implementation Phases

### Phase 1: Core Engine (done)
1. `pyproject.toml` with uv, llama-cpp-python
2. `LLMProvider` protocol + `LlamaCppProvider` (GGUF)
3. `WordMixer` — prompt LLM for combine/subtract, cache results
4. Tests with fake LLM provider (10 passing)

### Phase 2: FastAPI + WebSocket Game Server
5. FastAPI app with lifespan (load LLM once at startup)
6. `GameSession` state machine (LOBBY → COUNTDOWN → PLAYING → FINISHED)
7. `SessionManager` for creating/joining games
8. WebSocket endpoint + message dispatcher
9. Seed word list + target generation
10. Timer logic (asyncio.Task)
11. `POST /api/mix` debug endpoint

### Phase 3: React Frontend
12. Vite + React + TS scaffold with Tailwind 4
13. Zustand stores (gameStore, connectionStore)
14. WebSocket hook with reconnect + exponential backoff
15. WordTile + Workspace (drag-and-drop via @dnd-kit)
16. SubtractZone (two-slot drag interface)
17. Inventory panel, target panel, scoreboard
18. Animations (Framer Motion: merge, discovery, countdown)
19. Lobby screen

### Phase 4: Multiplayer
20. Matchmaking queue
21. Opponent state broadcast (throttled 1/sec)
22. Versus scoring (first-to-find bonus)
23. Results comparison screen

## Verification

1. **Engine**: `uv run python -c "from wmw.engine.mixer import WordMixer; ..."` — test word combinations
2. **API**: `curl localhost:8000/api/mix -d '{"word_a":"fire","word_b":"water","op":"add"}'`
3. **WebSocket**: Connect via browser devtools, send combine messages
4. **Frontend**: Open browser, drag words onto each other, verify animations
5. **Tests**: `uv run pytest` for backend
6. **Multiplayer**: Open two browser tabs, matchmake, verify both see updates
