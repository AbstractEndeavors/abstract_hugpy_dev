---
name: vi-ui-workbench-decomposition
description: Decomposition pattern that worked for a tightly-coupled frontend feature in video_intelligence_ui
metadata:
  type: feedback
---

For a tightly-coupled frontend feature in `video_intelligence_ui` (markup ↔ CSS ↔
routing/registry all change together), decompose as: ONE implementation subagent with
an airtight self-contained brief does all the code as a coherent unit; a SEPARATE
ops subagent does deploy+verify (rsync/curl — distinct skill, fresh context); the
orchestrator writes the docs directly at the end (docs need the final bundle hash and
benefit from held design context).

**Why:** splitting the sidebar / CSS / routing across parallel agents forces you to
pin exact class-name and markup contracts up front anyway, then eat integration
friction — a net loss. The 2026-07-03 workbench restructure (tabbed Studio + static
media-library sidebar) went clean this way: impl agent green on first `npm run build`,
deploy agent verified all 5 routes 200. Verified-good pattern, not a correction.
RE-VALIDATED same day on a follow-up (hoist sidebar to shell-level over ALL routes
incl. Generate + auto-add every artifact to the library on job completion + a tiny
composerBridge seam so the sidebar injects prompt parts on the Generate route): same
1 impl + 1 deploy + orchestrator-docs split, impl green first build, deploy clean.
Tip proven again: to make shell-wide state (a sidebar, a toggle) static across every
route, own it in `StationShell` ABOVE `<Routes>` and de-nest the per-route component.

**How to apply:** reach for this shape whenever a `video_intelligence_ui` change spans
components + `app.css` + `stations/registry.ts`/`StationShell.tsx` together. Give the
impl agent the exact as-built facts (mediaLibrary API surface, station render roots,
CSS token/grid primitives, config helpers, contract shapes) so it doesn't re-derive,
plus explicit deploy-marker strings to embed so the ops agent can grep the bundle.

**Refinement — cross-arm ports add a recon phase (2026-07-03, "fold Generate into
tabs + adopt sitewide navbar"):** when a change PORTS a component from a sibling arm
(e.g. copy the shared `Navbar` from `media_intelligence_ui` into `video_intelligence_ui`),
prepend a parallel READ-ONLY recon PAIR before the single impl agent: (1) source-arm
component recon — resolve the barrel to the real file, dump verbatim source+CSS+assets,
and CLASSIFY every import so you can rule "wholesale copy vs. lean faithful port"; (2)
target-arm as-built recon — exact registry/routing/StationShell/app.css facts. Feeding
both verbatim into ONE impl brief kept it green on first build again. The media `Navbar`
was already dep-clean (only `hugpyConfig.siteUrl` + a tiny `isEmbedded` URL helper, no
chat/router deps) → lean port: copy 2 tsx + 2 css + 1 png into `src/nav/`+`src/assets/`,
NO cross-arm import, NO new deps. CSS gotcha: the media nav's link colors resolve against
CSS custom-props defined on a `.hugpy-arm` WRAPPER with no literal fallbacks — make the
port self-contained by moving those vars onto `.hugpy-navbar` itself (else links render
unset/invisible). The Generate fold was a ONE-LINE registry change (`group:"studio"` on
the `generate` entry) that cascaded through the three group consumers — always check what
a `group`/filter field drives (NAV_SECTIONS, `STUDIO_STATIONS` filter, `ungrouped` route
loop) before assuming a bigger edit. Deploy note: this pass changed BOTH js+css hashes
(h3yot71G→AyWXgElg, nlfi_lix→dJJ-NL00) — the "diff dist/index.html refs before archiving"
rule in [[vi-ui-frontend-deploy]] handles that; archive BOTH superseded names, leave the
new png + `.claude/`.

**Re-validated + extended (2026-07-03, "always-expanded stations + viewport heights +
mobile hamburger", bundle `index-DmrNkE-o.js`/`index-CNiekhe8.css`):** same recon-pair →
1 impl → 1 deploy → orchestrator-docs shape held for a LARGER unit — ONE impl agent
cleanly did an 8-file change (4 stations + `StationShell` + `WorkbenchStation` + `app.css`
+ a NEW shared `src/video/DropReceptacle.tsx`) green on first `npm run build`. Confirms:
don't split a change just because it's big; split only when the pieces are genuinely
independent. The two asks (un-gate stations w/ a shared drop-receptacle; uniform
viewport-height chain + mobile drawer) both churned the SAME `app.css` + station panels,
so bundling them was correct.
- **CROSS-ARM MIRROR GOTCHA (media→video):** the media arm (`media_intelligence_ui`) is
  TAILWIND — inline utility classes (`md:`/`max-md:`, `text-token-*`, `-translate-x-full`).
  The video arm authors LITERAL hand CSS in one `src/style/app.css` (`.vi-*`). When you
  port a UX pattern across them, mirror the SEMANTICS (off-canvas `position:fixed` +
  `translateX` + scrim + z-index), NOT the classes — brief the impl agent explicitly or it
  may paste Tailwind. And use the TARGET arm's OWN breakpoint (video pivots its whole
  layout at 62rem) rather than the source's (media = 768px `md:`) so the arm stays
  internally consistent.
- **Media-arm mobile hamburger anatomy (saves recon next time):** hamburger = 3-line
  `menu` glyph in `chat/src/ui/ThreadHeader.tsx` toggling a `sidebarCollapsed` bool
  (init `window.innerWidth<768` in `chat/src/main.tsx`); drawer = `chat/src/ui/Sidebar.tsx`
  `<aside>` w/ `max-md:fixed inset-y-0 left-0 z-40` + `-translate-x-full`↔`translate-x-0`;
  scrim = a `fixed inset-0 z-30 bg-black/40 md:hidden` button in `main.tsx` that closes on
  tap. The media arm ALSO has a reusable drag/drop receptacle at
  `src/receptacle/src/ui/stages/FileInputStage.tsx` (`onDragOver/Leave/Drop` +
  `dataTransfer.files` + hidden-input-ref click, `.dragActive` class) — a clean reference
  for any "browse or drop" component.
- **Uniform viewport-height recipe for this arm (flex chain, no magic calc):** `.vi-shell
  {height:100dvh;flex-col}` → `.vi-main{flex:1;min-height:0;overflow:hidden}` →
  `.vi-workbench{height:100%;min-height:0;align-items:stretch}` → `.vi-workbench-main
  {flex-col;min-height:0;height:100%}` with `.vi-subtabs{flex-shrink:0}` (fixed row) +
  `.vi-workbench-body{flex:1;min-height:0;overflow-y:auto}` (scroll container) + sidebar
  `{height:100%;min-height:0;overflow:auto}`. `min-height:0` on every flex ancestor is the
  load-bearing bit. Keep any horizontal strip (`.vi-frames-grid`) on `overflow-x:auto`.

**Re-validated for a STORE-REFACTOR feature (2026-07-03, "session job tracker that
survives tab switches + reloads", bundle `index-DmrNkE-o.js`/`index-CNiekhe8.css` →
`index-DpLofb60.js`/`index-uvz1wgvu.css`):** same recon → 1 impl → 1 deploy →
orchestrator-docs shape. ONE impl agent cleanly did: a NEW module-level store
`src/video/jobTracker.ts` (mirrors `mediaLibrary.ts`) + refactored all FIVE station job
hooks onto it + a new sidebar Processes section + `app.css` — 8 files, green on first
`npm run build`. Confirms the rule again: a store and its consumer hooks are ONE unit —
do NOT split the store from the code that rides it. Give the impl agent the exact
mediaLibrary pattern facts + each hook's public-API return shape + the poll-cap
constants/string so it preserves contracts instead of re-deriving.
- **Durable/"survives-unmount" features → move the terminal side-effect OUT of the hook.**
  The bug was: switching sub-tabs unmounts the station → its hook's private `setInterval`
  dies → outputs never reach the library because the unmounted hook's `done`-branch never
  fires. FIX pattern: one module-level poll loop in the store polls ALL jobs regardless of
  mount, and the library-push moves into the store's terminal handler (the ONLY push site;
  `mediaLibrary` uri de-dup is the safety net). Any per-hook `setInterval` in this arm that
  also does a terminal `addToLibrary` has this latent amnesia bug.
- **`useSyncExternalStore` footgun (this arm uses it for both stores):** `getSnapshot` MUST
  return a referentially-stable cached value (only a new ref on real mutation) — mediaLibrary
  already does this. NEVER filter inside getSnapshot (returns a fresh array each call →
  "getSnapshot should be cached" → infinite render loop). Subscribe to the WHOLE list; filter
  per-consumer with `useMemo` in the hook body.
- **Poll seam for the durable job bus is under the `/api` base** (`jobStatusUrl` uses
  `apiBase` = `dev.hugpy.ai/api`), NOT the bare `/video/` SPA path (that returns the index
  shell). A `GET /api/video/jobs/<unknown-id>` returns a graceful `{job_id,result:null,
  status:null}` at HTTP 200 (no 500) — that's the seam a tracker's "expired/unknown job"
  path rides; headless e2e can enqueue a `frame_extract` via `/tmp/vi_e2e.py` shapes and
  watch queued→done.

**Extended to a FULL-STACK feature (2026-07-03, "generate_scene: one query -> N-frame
consecutive scene + assembled mp4", frontend bundle `index-mtHG1DJ6.js`/`index-DM-ij-jO.css`):**
the recon->impl->deploy->orchestrator-docs shape scales to backend+frontend by PAIRING each
phase and locking the JSON contract up front. Shape that went clean: (0) parallel recon PAIR
— one general-purpose agent for the backend rails (+ the decisive coherence investigation:
grep the managers plane for an img2img `(framework,task)` pair; verdict was seed+prompt v1
because only `("transformers","text-to-image")` is registered), one Explore agent for the
frontend as-built; (1) parallel impl PAIR — a backend agent (schema+runner+shared-guard
refactor+registries+route+selftest) and a frontend agent (mode switch + tracker kind + poll
cap + result view), BOTH handed the SAME verbatim locked contract so they never diverge;
(2) parallel deploy PAIR — a backend ops agent (VM selftest pre-gate + one restart + live
e2e, see [[vi-backend-deploy-gates]]) and a frontend ops agent (rsync/archive/curl, see
[[vi-ui-frontend-deploy]]); (3) orchestrator writes ROADMAP + map §12.x directly. Both impl
agents green on FIRST gate, both deploys clean, full e2e ALL PASS. **The load-bearing move
is CONTRACT-LOCK:** pin the exact request/response JSON (field names, output tuple order —
here N image refs then the video ref LAST) in both impl briefs verbatim, so backend and
frontend run in parallel with zero integration friction. The design brief pre-specified the
contract, so recon only had to CONFIRM the coherence verdict + capture as-built shapes to
mirror. Don't split backend from frontend into serial phases when the contract is knowable
up front — parallelize behind the lock.

**Extended again to an INFERENCE-PLANE + FLEET full-stack feature (2026-07-03, "make image
conditioning REAL: img2img start-frame + true frame[i+1]=img2img(frame[i]) chaining", bundle
`index-BwCqe9va.js`/`index-BFC06I-3.css`):** the same recon→contract-lock→impl-pair→deploy
shape held for a change that reached the managers plane AND a GPU worker, with TWO structural
refinements worth reusing:
- **Recon TRIO, not pair, when a fleet worker is in scope.** Parallel read-only recon: (a)
  backend managers-plane change-set (how `(framework,task)` runner/builder/model-config/
  validate_registry work → the precise additive edit list), (b) WORKER recon (SSH to op:
  install method, deploy lever, pip-freeze snapshot for rollback), (c) frontend as-built. The
  worker recon is what surfaced the STOP boundary (op = pinned PyPI wheel → a real release).
  Feeding all three verbatim into the contract-lock kept both impl agents green on first gate.
- **Contract-lock can encode a "held" decision.** When the fleet can't serve the new task yet
  (pinned-wheel worker), the locked contract said: register the runner/builder (inert), but
  HOLD the model's task-advertisement flip, and make the runner FAIL HONESTLY (retryable
  JobError) via an availability probe. Both impl agents built to that; live e2e asserts the
  honest error, and a VM CPU selftest proves the real generation. See
  [[vi-img2img-fleet-release]] for the full posture.
- **Add a DIAGNOSTIC phase when a deploy agent reports an anomaly.** The backend verify agent
  blamed my registry rows for a multi-GB model prefetch; a dedicated diagnostic agent proved
  it EXTERNAL (another terminal's download campaign). ALWAYS root-cause an anomaly (find the
  trigger file:line or external process id) before "fixing" — and it still isolated a real,
  separable one-line scope trim I DID own (keep the new task out of `HF_TASK_TO_TASKS` so
  discovery doesn't auto-advertise unvetted models). Treat every subagent claim as a draft.

**Gotchas learned:** the `mediaLibrary.ts` store already had the live subscribe seam
(`subscribeLibrary` + `useMediaLibrary` on `useSyncExternalStore`) — verify what
exists before briefing "add a seam". The docs asked for map "§12.5" but the map only
had §12.1–12.3; I added §12.4 (next sequential) to avoid a numbering gap and flagged
it. The frontend contract type lives at `src/video/contract.ts` (NOT `src/stations/`).
See [[vi-ui-frontend-deploy]] for the deploy/archive landmine (and its 2026-07-03
docroot/mount-namespace anomaly note).
