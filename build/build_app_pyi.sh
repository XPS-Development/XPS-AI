cd "$(dirname "$0")"
pyinstaller XPS-AI.spec --workpath pyi_build --distpath pyi_dist
