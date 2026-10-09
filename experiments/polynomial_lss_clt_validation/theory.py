from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


STATISTICS = ("hoeffding_d", "bkr_r", "bdy_tau_star")
POWERS = (2, 4)

DISPLAY_NAMES = {
    "hoeffding_d": "Hoeffding's D",
    "bkr_r": "BKR R",
    "bdy_tau_star": "BDY tau*",
}


@dataclass(frozen=True)
class LimitParameters:
    center: float
    mean: float
    variance: float

    def standardize(self, value: float) -> float:
        return (value - self.center - self.mean) / sqrt(self.variance)


def finite_center(statistic: str, power: int, n: int, p: int, q_n: float) -> float:
    q = q_n
    if statistic == "hoeffding_d" and power == 2:
        return p * (1.0 + q / 9.0) + (5.0 * n - 8.0) * q**2 / 9.0
    if statistic == "bkr_r" and power == 2:
        return p * (1.0 + q) + 2.0 * (n - 1.0) * q**2
    if statistic == "bdy_tau_star" and power == 2:
        return p * (1.0 + 9.0 * q / 25.0) + 3.0 * (7.0 * n - 12.0) * q**2 / 25.0

    if statistic == "hoeffding_d" and power == 4:
        bulk = p * (1.0 + 2.0 * q / 3.0 + 14.0 * q**2 / 81.0 + q**3 / 81.0)
        correction = 2.0 * q**2 / 81.0 * (
            n * (2.0 * q**2 + 28.0 * q + 135.0)
            + 2.0 * q**2
            + 88.0 * q
            - 216.0
        )
        return bulk + correction
    if statistic == "bkr_r" and power == 4:
        bulk = p * (1.0 + 6.0 * q + 6.0 * q**2 + q**3)
        correction = 4.0 * q**2 * (
            n * (q**2 + 5.0 * q + 3.0) + q**2 + 2.0 * q - 3.0
        )
        return bulk + correction
    if statistic == "bdy_tau_star" and power == 4:
        bulk = p * (
            1.0 + 54.0 * q / 25.0 + 702.0 * q**2 / 625.0 + 81.0 * q**3 / 625.0
        )
        correction = 18.0 * q**2 / 625.0 * (
            n * (18.0 * q**2 + 132.0 * q + 175.0)
            + 18.0 * q**2
            + 36.0 * q
            - 300.0
        )
        return bulk + correction
    raise ValueError(f"unsupported statistic/power pair: {statistic}, {power}")


def limiting_mean(statistic: str, power: int, q: float) -> float:
    if statistic == "hoeffding_d" and power == 2:
        return -q / 9.0
    if statistic == "bkr_r" and power == 2:
        return -q
    if statistic == "bdy_tau_star" and power == 2:
        return -9.0 * q / 25.0
    if statistic == "hoeffding_d" and power == 4:
        return -q * (98.0 * q**2 + 493.0 * q + 882.0) / 1323.0
    if statistic == "bkr_r" and power == 4:
        return -3.0 * q * (98.0 * q**2 + 101.0 * q + 98.0) / 49.0
    if statistic == "bdy_tau_star" and power == 4:
        return -27.0 * q * (882.0 * q**2 + 2085.0 * q + 2450.0) / 30625.0
    raise ValueError(f"unsupported statistic/power pair: {statistic}, {power}")


def limiting_variance(statistic: str, power: int, q: float) -> float:
    if statistic == "hoeffding_d" and power == 2:
        return 1060.0 * q**2 / 3969.0
    if statistic == "bkr_r" and power == 2:
        return 1060.0 * q**2 / 49.0
    if statistic == "bdy_tau_star" and power == 2:
        return 17172.0 * q**2 / 6125.0
    if statistic == "hoeffding_d" and power == 4:
        polynomial = (
            530.0 * q**4
            + 7132.0 * q**3
            + 33493.0 * q**2
            + 64188.0 * q
            + 42930.0
        )
        return 8.0 * q**2 * polynomial / 35721.0
    if statistic == "bkr_r" and power == 4:
        polynomial = (
            530.0 * q**4
            + 2892.0 * q**3
            + 4965.0 * q**2
            + 2892.0 * q
            + 530.0
        )
        return 72.0 * q**2 * polynomial / 49.0
    if statistic == "bdy_tau_star" and power == 4:
        polynomial = (
            42930.0 * q**4
            + 348732.0 * q**3
            + 943461.0 * q**2
            + 968700.0 * q
            + 331250.0
        )
        return 5832.0 * q**2 * polynomial / 19140625.0
    raise ValueError(f"unsupported statistic/power pair: {statistic}, {power}")


def limit_parameters(
    statistic: str, power: int, n: int, p: int, q_n: float
) -> LimitParameters:
    return LimitParameters(
        center=finite_center(statistic, power, n, p, q_n),
        mean=limiting_mean(statistic, power, q_n),
        variance=limiting_variance(statistic, power, q_n),
    )
