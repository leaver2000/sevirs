import ast
import sys


def main() -> int:
    with open("sevirs/mrms/constants.py", "r") as f:
        x = ast.parse(f.read())
    _all_ = []
    for item in x.body:
        if isinstance(item, (ast.ClassDef, ast.FunctionDef)) and not item.name.startswith("_"):
            _all_.append(item.name)
        elif isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    _all_.append(target.id)
        else:
            print(ast.dump(item))
    print(_all_)
    return 0


if __name__ == "__main__":
    sys.exit(main())
