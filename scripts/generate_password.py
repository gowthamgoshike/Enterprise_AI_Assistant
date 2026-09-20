from getpass import getpass
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.auth_service.password_service import hash_password  # noqa: E402


def main() -> None:
    password = getpass(
        "Password: "
    )

    print(
        hash_password(password)
    )


if __name__ == "__main__":
    main()
