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

# Repository rules

- Keep the root `README.md` describing how the app works as of the current commit. Any commit that changes behavior, interfaces, or data must update the "How it works right now" section and add a line to its changelog in the same commit.
