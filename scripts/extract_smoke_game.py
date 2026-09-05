"""Extract ONE solvable game from the official release ZIP for local adapter smoke tests.

No archive-controlled paths are used as output paths. This does not prepare the
full training dataset and does not overwrite an existing destination file.
"""

import argparse
import json
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("output", help="New output path ending in .tw-pddl")
    args = parser.parse_args()
    output = Path(args.output)
    if output.suffix != ".tw-pddl":
        parser.error("Output must end in .tw-pddl")
    with zipfile.ZipFile(args.archive) as archive:
        for name in sorted(archive.namelist()):
            if "/train/pick_and_place_simple-" not in name or not name.endswith("/game.tw-pddl"):
                continue
            if archive.getinfo(name).file_size > 5_000_000:
                raise ValueError("Unexpectedly large game entry")
            data = archive.read(name)
            if not json.loads(data).get("solvable"):
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("xb") as stream:
                stream.write(data)
            print(json.dumps({"source_member": name, "output": str(output.resolve())}))
            return
    raise ValueError("No solvable pick-and-place game found")


if __name__ == "__main__":
    main()
