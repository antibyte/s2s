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

## Speech Lab Runtime Progress

- Module installation status reports model bytes and Docker image bytes separately. The controller deduplicates pull events by layer and reports both increasing transferred bytes and the observed layer total while the pull is active. The gateway prefers that observed total over the catalog estimate in `/api/v1/modules/{backend_id}/install`.
- The Browser Lab updates its model-plus-image total when Docker reports actual layer sizes, clamps visible progress so newly discovered layers or out-of-order WebSocket and polling updates cannot move the bar backward, and shows transferred image bytes during image preparation. Catalog image sizes remain estimates until a pull reports its layers.
- The managed Qwen3-TTS Vulkan image builds the pinned native `qwentts.cpp` server with Vulkan and opens the already provisioned 0.6B talker and codec GGUF files from `/models`; do not use the Python GGML wrapper without its optional native binding.
- Verify changes with `go test ./...` in `s2s-vulkan/controller`, `cargo check --tests`, and `node --check s2s-vulkan/web/app.js`.
- For the Vulkan sidecar, build `s2s-vulkan/docker/Dockerfile.tts-vulkan` and run `tts-server --help` inside the resulting image before publishing it.

## Speech Lab Response Language

- The bundled Granite LLM writes the Browser Lab's BOT text; Qwen3-TTS voices only synthesize that text. With the default German lab language and default system prompt, send the tested German-specific prompt to Granite for each turn. Preserve explicitly configured `S2S_SYSTEM_PROMPT` values and the generic prompt for other language settings. The LLM benchmark must use the same effective prompt as live turns.
- Verify German answer content with the questions `Dein Name? Wie ist dein Name?` and `Sprich bitte Deutsch mit mir.` through the bundled LLM, plus an English and custom-prompt regression check. A new WebSocket session is needed to test a newly published gateway image.
