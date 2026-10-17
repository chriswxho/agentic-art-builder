# Picture-agent rules

These rules apply to any agent or subagent creating pictures in `drawing-sandbox`.

- Build images only from the Drawing Sandbox primitives: paths, polygons, and ellipses. Groups and motif calls may assemble those primitives.
- Use a white (`#ffffff`) canvas and only cyan (`#00ffff`), magenta (`#ff00ff`), yellow (`#ffff00`), or black (`#000000`) paint.
- Every visible mark must use `opacity: 0.5`.
- Keep each transformed primitive at or below 96 canvas units in span and each transformed stroke at or below 12 canvas units wide.
- Create detail by overlaying and combining small marks. Do not bypass the limits with a large background-colored shape or a pre-rendered bitmap.
- Use the agent-facing renderer, which enforces these constraints, before accepting an image.
- Record each meaningful progress state with a stable run ID, the complete prompt set, the composition/layer plan, and a step label. Mark the accepted output as final so its PNG, SVG, and scene JSON are preserved alongside the progress frames.

The machine-readable policy lives in `drawing-sandbox/drawing_sandbox/policy.py`. The recording API is `POST /api/record`.

## Nemotron iteration handoffs

- Consecutive Nemotron picture agents receive a `PictureContext` JSON sidecar conforming to `drawing-sandbox/drawing_sandbox/proto/agent_iteration.proto`.
- Each agent is one forward-inference call to Crusoe Managed Inference using `nvidia/Nemotron-3-Nano-Omni-Reasoning-30B-A3B` and must return exactly one typed `AgentProposal` skill: `split_layers`, `combine_layers`, `add_objects`, or `remove_objects`.
- Agents must not edit renderer scenes or context files directly. The orchestrator validates the context hash and revision, applies the proposal atomically, renders it through the enforced policy, records it, and only then publishes the next context.
- Preserve stable layer and object IDs from the context. New object and layer IDs must be unique and descriptive.
- Use the current rendered PNG and reference assets listed in the context when critiquing a revision.
- Record the exact text prompts, model identifier, response identifier, token usage, raw response, and accepted proposal for every inference. Never record the API key or image data URLs.
