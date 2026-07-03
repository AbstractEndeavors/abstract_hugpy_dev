---
name: vi-ui-frontend-deploy
description: video_intelligence_ui (/video arm) frontend build+deploy recipe, the gate, and the .claude-in-assets archive landmine
metadata:
  type: reference
---

The hugpy Video Intelligence UI source is `dev/video_intelligence_ui/` (Vite +
Tailwind v4, React 18, react-router-dom v7). Live at `https://dev.hugpy.ai/video/`.

**Deploy recipe (frontend-only, no service restart needed):**
1. `npm run build` (node v20 via nvm) → emits `dist/index.html` +
   `dist/assets/index-<hash>.js|css` (base `/video/`).
2. rsync ADDITIVE (never `--delete`): `dist/` → `dev/abstract_hugpy_dev/src/abstract_hugpy_dev/console_dist/video/`.
3. Move the SUPERSEDED `index-*.js|css` pair from `console_dist/video/assets/`
   into `console_dist/video/assets_archive/`. Keep only the live pair in `assets/`.
4. Webpack `:7001` serves `dist/` directly (public dev.hugpy.ai pages); gunicorn
   `:7002` serves `console_dist/video/`. Two docroots — the rsync covers the second.

**LANDMINE:** `console_dist/video/assets/` also contains a `.claude/` dir — it holds
THIS agent-memory system (`.claude/agent-memory/task-orchestrator/`). When archiving,
move ONLY `index-*.js` / `index-*.css` BY EXACT NAME. Never `mv assets/* archive/`
or any wildcard that could sweep `.claude/` (or the new live pair) into the archive.

**Archive only what ACTUALLY changed, not a "pair".** Vite hashes each asset by its
own content, so a CSS-only-unchanged rebuild emits a NEW `index-<hash>.js` but the
SAME `index-<hash>.css` as before (e.g. 2026-07-03 follow-up: JS `CCvxOdfx`→`h3yot71G`,
CSS `nlfi_lix` identical). In that case archive ONLY the superseded JS by exact name
and LEAVE the css (it's still live). Diff `dist/index.html`'s asset refs against the
target's before deciding what to archive — blindly moving the "css pair" would banish
the still-live stylesheet. rsync is additive (no `--delete`), so the stale file just
sits alongside until you move the exact superseded one.

**Build gate:** the gate is `npm run build` exit 0 with no warnings. Project-wide
`tsc --noEmit` is RED at baseline (missing react-jsx/vite-client type wiring) — don't
chase it; only avoid NEW tsc errors in files you author.

**Verify live:** curl `/video/`, `/video/image-crop`, `/video/audio-crop`,
`/video/frames`, `/video/generate` → all 200 (SPA). Confirm served `/video/` HTML
references the new hash, and the served JS carries the poll invariant
`Still running on the server — check again.` SPA users must HARD-RELOAD for a new bundle.

**⚠ OBSERVED ANOMALY (2026-07-03, needs human confirmation) — docroot + mount-namespace
divergence + a memory wipe.** After the img2img deploy: the spawned deploy agent followed
this recipe (rsync `dist/` → `console_dist/video/`, archive superseded by exact name) and
VERIFIED the new bundle `index-BwCqe9va.js` live via public curl of `dev.hugpy.ai/video/`
(HTML on the new hash, invariant string present) — so the frontend WAS correctly served.
BUT in the ORCHESTRATOR's own post-run filesystem view, the live `index-BwCqe9va.js`
physically resides at `dev/ui/dist/video/assets/` (the canonical `hugpy/ui` tree from the
repo revamp), while `console_dist/video/assets/` still held a STALE bundle
(`index-Ccc6SYdU.js`), had NO `assets_archive/`, and its `.claude/agent-memory/` had been
wiped (3 memory files + MEMORY.md gone mid-conversation; restored from context). Strong
evidence that SUBAGENTS run in a different mount namespace than the orchestrator, where
`console_dist/video/` maps to the served docroot (likely `dev/ui/dist/video/`). Timestamps
on this mount are unreliable (a just-written file showed a stale mtime). TAKEAWAYS: (a) the
real served docroot may now be `dev/ui/dist/video/`, not `console_dist/video/` — verify
before trusting step 2/4 above; (b) treat this agent-memory dir as loss-prone — a wipe
recurred here; (c) the public-curl verification is the source of truth for "is it live",
not the local bytes. See [[vi-img2img-fleet-release]].

See also [[vi-ui-workbench-decomposition]]. Deploy details also live in the repo:
`dev/video_intelligence_ui/ROADMAP.md` (Phase 8) and `hugpy_video_intelligence_map.md` §12.4.
