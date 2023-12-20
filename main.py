#!venv/bin/python3
# -*- coding: utf-8 -*-

# Модуль для работы с пресетом данных
# Для поиска выбросов используется метод
# Инициализируемому объекту класса передается путь к пресету данных

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class DataSet:
    def __init__(self, path: str) -> pd.DataFrame:
        self.df = pd.read_csv(path, parse_dates=["time_index"])

    def __str__(self) -> str:
        return self.df.info()

    def interquartile_range_outliers(
        self, Q1: float = 0.25, Q3: float = 0.75, whis: float = 1.5, top: bool = False
    ) -> None:
        """Метод для визуализации выбросов по межквантильному размаху. Можно задать произвольный первый и третий квантиль

        Args:
            Q1 (float, optional): Укажи первый квантиль. Defaults to 0.25.
            Q3 (float, optional): Укажи верхний квантиль. Defaults to 0.75.
            whis (float, optional): Интерквартильное расстояние. Defaults to 1.5.
            :param bool top: отмеччать выбросами верхние значения, defaults to False
        """
        # Сделаем ящики с усами для всех series_id
        fig, axs = plt.subplots(figsize=(20, 5))
        axs.boxplot(
            [self.df["values"][self.df["series_id"] == n] for n in range(1, 11)],
            whis=whis,
        )
        axs.set_xticks([x for x in range(1, 11)], [f"Ряд {n}" for n in range(1, 11)])

        def outlier(row: pd.DataFrame, up: float, low: float, top):
            if top:
                return 1 if (row["values"] > up) | (row["values"] < low) else 0
            else:
                return 1 if (row["values"] < low) else 0

        # Сделаем графики с выбросами для всех series_id
        fig, axes = plt.subplots(nrows=10, ncols=1, figsize=(20, 40))
        for n in range(1, 11):
            data = self.df[self.df["series_id"] == n]

            _Q1 = data["values"].quantile(q=Q1)
            _Q3 = data["values"].quantile(q=Q3)
            IQR = _Q3 - _Q1

            low = _Q1 - whis * IQR
            up = _Q3 + whis * IQR

            data.loc[:, ["test_outlier"]] = data.apply(
                outlier, low=low, up=up, top=top, axis=1
            )

            try:
                outl_count = data.value_counts("test_outlier")[1]
            except:
                outl_count = 0

            (line1,) = axes[n - 1].plot(
                data["time_index"], data["values"], label=f"Значение ряда {n}"
            )
            (line2,) = axes[n - 1].plot(
                data["time_index"][data["test_outlier"] == 1],
                data["values"][data["test_outlier"] == 1],
                "rx",
                label="Выбросы (Всего: %d)" % outl_count,
            )
            axes[n - 1].legend(handles=[line1, line2])

    def std_dev_outliers(self, std: int = 3, top: bool = False) -> None:
        """
        Метод для визуализации выбросов z-оценки. Рисует два графика рядом: Гистограмму с отсечкой выбросов верхней и нижней границы и линейный график с значениями.

        :param int std: граница средних отклонений, defaults to 3
        :param bool top: отмеччать выбросами верхние значения, defaults to False
        """
        fig = plt.figure(figsize=(20, 50))
        grid = GridSpec(10, 10, figure=fig)

        for i in range(10):
            data = self.df[self.df["series_id"] == i + 1]
            data.loc[:, ["std_data"]] = (
                data["values"] - np.mean(data["values"])
            ) / np.std(data["values"])

            if top:
                x = data.loc[:, ["time_index"]][
                    (data["std_data"] < -std) | (data["std_data"] > std)
                ]
                y = data.loc[:, ["values"]][
                    (data["std_data"] < -std) | (data["std_data"] > std)
                ]
            else:
                x = data.loc[:, ["time_index"]][(data["std_data"] < -std)]
                y = data.loc[:, ["values"]][(data["std_data"] < -std)]

            ax = fig.add_subplot(grid[i, 1:3])
            ax.hist(data["std_data"], bins=20)
            ax.axvline(3, color="r", linestyle="dashed", linewidth=2)
            ax.axvline(-3, color="r", linestyle="dashed", linewidth=2)

            ax2 = fig.add_subplot(grid[i, 3:])

            ax2.plot(data["time_index"], data["values"], label=f"Значение ряда {i+1}")
            ax2.plot(x, y, "rx", label="Выбросы (Всего: %d)" % len(y))
            ax2.legend()
