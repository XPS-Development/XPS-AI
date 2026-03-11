from datetime import datetime
from pathlib import Path
from traceback import format_exception


def save_error_dump(exc: BaseException) -> Path:
    """
    Save an error dump with full traceback to the error_dumps folder.

    The dump file is created under ``Path.cwd() / "error_dumps"`` with a
    timestamped name. The file contains the exception type, message, and
    full traceback.

    Parameters
    ----------
    exc : BaseException
        Exception instance to serialize.

    Returns
    -------
    Path
        Path to the written dump file.
    """
    dumps_dir = Path.cwd() / "error_dumps"
    dumps_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"error_{timestamp}.txt"
    dump_path = dumps_dir / filename

    lines = format_exception(exc)
    dump_text = "".join(lines)

    header = [
        f"Exception type: {type(exc).__name__}",
        f"Exception message: {exc}",
        "",
        "Traceback:",
        "",
    ]
    dump_path.write_text("\n".join(header) + dump_text, encoding="utf-8")
    return dump_path
