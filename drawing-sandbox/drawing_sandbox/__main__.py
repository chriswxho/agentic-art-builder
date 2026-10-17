from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


def export(scene,folder):
    from .policy import composition_constraints
    from .renderer import check_scene,render,to_svg
    checks=check_scene(scene,composition_constraints())
    if not checks.valid:
        raise ValueError("; ".join(checks.violations))
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    (folder/"scene.json").write_text(json.dumps(scene,indent=2,allow_nan=False))
    (folder/"drawing.svg").write_text(to_svg(scene))
    render(scene).save(folder/"drawing.png")
    (folder/"checks.json").write_text(json.dumps(asdict(checks),indent=2))
    return checks


def main(argv=None):
    parser=argparse.ArgumentParser(description="Assemble → render → inspect → revise")
    sub=parser.add_subparsers(dest="command")
    viewer=sub.add_parser("serve",help="Open the local drawing editor (default)")
    viewer.add_argument("--port",type=int,default=8766)
    viewer.add_argument("--no-browser",action="store_true")
    renderer=sub.add_parser("render",help="Render a scene JSON to SVG and PNG")
    renderer.add_argument("scene")
    renderer.add_argument("--out",default="output")
    presets=sub.add_parser("examples",help="Export the three starter drawings")
    presets.add_argument("--out",default="examples")
    checker=sub.add_parser("check",help="Check a drawing without rendering it")
    checker.add_argument("scene")
    checker.add_argument("--json",action="store_true",help="Print raw JSON instead of a readable summary")
    reporter=sub.add_parser("report",help="Write a readable report.md for a recorded run")
    reporter.add_argument("run_dir")
    args=parser.parse_args(argv)
    try:
        if args.command in (None,"serve"):
            from .server import serve
            serve(getattr(args,"port",8766),not getattr(args,"no_browser",False))
        elif args.command=="render":
            from .report import summarize_render
            scene=json.loads(Path(args.scene).read_text())
            checks=export(scene,args.out)
            print(summarize_render(scene,asdict(checks)))
            print("\nSaved to "+str(Path(args.out).resolve()))
        elif args.command=="examples":
            from .examples import examples
            for example in examples():
                export(example["scene"],Path(args.out)/example["id"])
                print(example["title"])
        elif args.command=="check":
            from .policy import composition_constraints
            from .renderer import check_scene
            from .report import summarize_render
            scene=json.loads(Path(args.scene).read_text())
            checks=check_scene(scene,composition_constraints())
            print(json.dumps(asdict(checks),indent=2) if args.json else summarize_render(scene,asdict(checks)))
            return 0 if checks.valid else 1
        elif args.command=="report":
            from .report import write_run_report
            print(str(write_run_report(args.run_dir).resolve()))
    except (ValueError,OSError) as error:
        print(str(error),file=sys.stderr)
        return 1
    return 0


if __name__=="__main__":
    raise SystemExit(main())
