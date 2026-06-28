import pandas as pd
from math import log


class VolSurface:
    def __init__(self, df):
        self.ticker = None 
        self.time = None 
        self.exchange = None
        self.df = df.copy()

    @classmethod
    def from_excel(cls, path, sheet_name=0):
        df = pd.read_excel(path, sheet_name=sheet_name)
        return cls(df)

    def add_coordinates(self):
        """
        Requires columns:
            maturity, strike, forward, iv
        """
        self.df["total_variance"] = self.df["iv"] ** 2 * self.df["maturity"]
        self.df["forward_moneyness"] = self.df["strike"] / self.df["forward"]
        self.df["log_forward_moneyness"] = (
            self.df["strike"] / self.df["forward"]
        ).apply(log)

        return self.df

    def iv_for_maturity(self, maturity):
        """
        Returns:
            {strike: iv}
        """
        sub = self.df[self.df["maturity"] == maturity]
        return dict(zip(sub["strike"], sub["iv"]))

    def iv_for_strike(self, strike):
        """
        Returns:
            {maturity: iv}
        """
        sub = self.df[self.df["strike"] == strike]
        return dict(zip(sub["maturity"], sub["iv"]))

    @property
    def iv_by_maturity(self):
        """
        Returns:
            {maturity: {strike: iv}}
        """
        return {
            maturity: dict(zip(group["strike"], group["iv"]))
            for maturity, group in self.df.groupby("maturity")
        }

    @property
    def iv_by_strike(self):
        """
        Returns:
            {strike: {maturity: iv}}
        """
        return {
            strike: dict(zip(group["maturity"], group["iv"]))
            for strike, group in self.df.groupby("strike")
        }