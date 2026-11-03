"""Исключения 64vofcf32. Все наследуются от VofcfError."""


class VofcfError(Exception):
    """Базовое исключение 64vofcf32."""
    pass


class QuantizationError(VofcfError):
    """Ошибка в процессе квантизации (несовместимый dtype, пустой тензор и т.п.)."""
    pass


class CalibrationError(VofcfError):
    """Ошибка калибровки: не удалось подобрать scale/zero_point."""
    pass


class UnsupportedSchemeError(VofcfError):
    """Неизвестная схема квантизации (per_tensor / per_channel / ...)."""
    pass


class QualityBelowThresholdError(VofcfError):
    """Оценка качества ниже заданного порога (MSE/MAE/cosine)."""
    pass


class BudgetError(VofcfError):
    """Не удалось подобрать dtype под заданный бюджет памяти."""
    pass


__all__ = [
    "VofcfError",
    "QuantizationError",
    "CalibrationError",
    "UnsupportedSchemeError",
    "QualityBelowThresholdError",
    "BudgetError",
]