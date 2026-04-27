import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from config import settings

security = HTTPBasic()


def require_auth(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    """HTTP Basic Auth с захардкоженным юзером/паролем из .env. На один пользователь
    этого достаточно; для большего нужен полноценный auth-провайдер."""
    correct_user = secrets.compare_digest(
        credentials.username.encode("utf-8"),
        settings.AUTH_USERNAME.encode("utf-8"),
    )
    correct_pass = secrets.compare_digest(
        credentials.password.encode("utf-8"),
        settings.AUTH_PASSWORD.encode("utf-8"),
    )
    if not (correct_user and correct_pass):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный логин или пароль",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username
