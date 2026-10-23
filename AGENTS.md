<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **s2s** (1337 symbols, 3484 relationships, 115 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> Index stale? Run `node .gitnexus/run.cjs analyze` from the project root — it auto-selects an available runner. No `.gitnexus/run.cjs` yet? `npx gitnexus analyze` (npm 11 crash → `npm i -g gitnexus`; #1939).

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows. For regression review, compare against the default branch: `detect_changes({scope: "compare", base_ref: "main"})`.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `query({search_query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `context({name: "symbolName"})`.
- For security review, `explain({target: "fileOrSymbol"})` lists taint findings (source→sink flows; needs `analyze --pdg`).

## Never Do

- NEVER edit a function, class, or method without first running `impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `rename` which understands the call graph.
- NEVER commit changes without running `detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/s2s/context` | Codebase overview, check index freshness |
| `gitnexus://repo/s2s/clusters` | All functional areas |
| `gitnexus://repo/s2s/processes` | All execution flows |
| `gitnexus://repo/s2s/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->

## Speech Lab runtime and release ownership

- `s2s-vulkan/config/backends.json` owns backend IDs, voice IDs/labels, protocols,
  immutable model artifacts and hardware variants. Confucius/Supertonic remain
  defaults; new entries never overwrite persisted stack/voice choices.
- `src/tts/phonon.rs` owns the native Phonon HTTP sidecar. Keep its 2,000-character,
  120-second, one-process, two-thread and 16-MiB limits, disconnect cancellation,
  WAV validation and private temporary cleanup. `config/phonon.json` pins every
  model, voice, runtime and release package. Do not put HF tokens in build assets.
- `scripts/prepare_phonon.py` verifies/extracts the existing AuraGo resources;
  Python is build-only. Phonon resources stay outside `/models` so model-volume
  mounts cannot hide them. Model weights retain CC BY-NC 4.0.
- Granite Speech uses the requested i1-Q4_K_M GGUF and the static Q8_0 projector,
  downloaded by the existing consent/hash/atomic-install path. Keep CPU/CUDA/
  Vulkan llama.cpp base images pinned by digest. Never silently change precision.
- Regenerate `deploy/speech-lab-bundle.json` only with
  `python scripts/speech_lab_bundle.py --write`, then run it without `--write`.
  Runtime image keys must also be recognized by AuraGo's managed-bundle adapter.
- `.github/workflows/ghcr.yml` owns GHCR and signed bundle-v2 publication. Stage
  checked Phonon archives in a draft release; publish only after uploading the
  checksum and raw Ed25519 signature. Never change live stack/chat settings as
  a side effect of publication.
- Workflow dispatch `target=gateway` updates only gateway image tags (including
  `latest` on main); it leaves sidecar images and published bundles untouched.
- Verify `cargo test --release --locked`, `go test ./...` in `controller/`,
  `python -m unittest discover -s tests -p test_speech_resources.py`, and real
  offline Phonon sidecar/gateway synthesis plus Granite CPU transcription.
  Report separately whether CUDA/Vulkan had physical hardware coverage.
