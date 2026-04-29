import re
from typing import Optional

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError, InvalidHash

from config import settings


# Параметры Argon2id рекомендованы OWASP:
# - 19 MiB памяти, 2 итерации, 1 поток. Время хеширования ~50-100 ms на CPU.
_hasher = PasswordHasher(
    time_cost=2,
    memory_cost=19 * 1024,
    parallelism=1,
    hash_len=32,
    salt_len=16,
)


# Регулярки для проверки силы пароля. Минимум: длина + буквы + цифры.
_RE_LETTER = re.compile(r"[A-Za-zА-Яа-яЁё]")
_RE_DIGIT = re.compile(r"\d")


# Топ-плохих паролей, чтобы быстро отсеять самое очевидное. Это НЕ полный
# список, но защищает от dump-пары "password123" и т.п. Для полноценной
# проверки можно подключить HIBP API, но это уже out of scope для MVP.
_COMMON_BAD = {
    "password", "qwerty", "12345678", "123456789", "1234567890",
    "qwerty123", "password1", "password123", "admin", "admin123",
    "letmein", "welcome", "iloveyou", "monkey", "dragon",
}


def hash_password(password: str) -> str:
    """Argon2id хеш с солью. Возвращает строку формата $argon2id$..."""
    return _hasher.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    """Проверяет пароль. False вместо исключения для нормальных «не совпало» —
    но True/False независимо от типа ошибки, чтобы не сливать timing-инфу."""
    try:
        _hasher.verify(hashed, password)
        return True
    except (VerifyMismatchError, VerificationError, InvalidHash):
        return False
    except Exception:
        return False


def needs_rehash(hashed: str) -> bool:
    """True, если параметры устарели и стоит перехешировать после успешного login."""
    try:
        return _hasher.check_needs_rehash(hashed)
    except Exception:
        return False


def validate_password_strength(password: str) -> Optional[str]:
    """Возвращает None если пароль ок, иначе строку с описанием проблемы."""
    if not isinstance(password, str):
        return "Пароль обязателен"
    if len(password) < settings.PASSWORD_MIN_LENGTH:
        return f"Пароль должен быть не короче {settings.PASSWORD_MIN_LENGTH} символов"
    if len(password) > 256:
        return "Пароль слишком длинный (максимум 256 символов)"
    if not _RE_LETTER.search(password):
        return "Пароль должен содержать хотя бы одну букву"
    if not _RE_DIGIT.search(password):
        return "Пароль должен содержать хотя бы одну цифру"
    if password.lower() in _COMMON_BAD:
        return "Этот пароль слишком распространён, выберите другой"
    return None
