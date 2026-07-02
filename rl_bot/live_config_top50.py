"""
Live config for 84 profitable categories from collector data training.
Generated from test split evaluation (last 2h holdout).
"""
from dataclasses import dataclass


@dataclass
class CategoryConfig:
    name: str
    max_inventory: int = 2
    quote_size: int = 1
    capital: float = 0.35


# Categories profitable on out-of-sample test split
TOP_50_CATEGORIES = [
    CategoryConfig(name="KXMVESPORTSMULTIGAMEEXTENDED"),  # pnl=$+32762.42 win=4771/7479
    CategoryConfig(name="KXMVECROSSCATEGORY"),  # pnl=$+8251.93 win=1296/1769
    CategoryConfig(name="KXWCSCORE"),  # pnl=$+604.91 win=69/71
    CategoryConfig(name="KXWCGOAL"),  # pnl=$+524.75 win=70/74
    CategoryConfig(name="KXWC1HSCORE"),  # pnl=$+502.01 win=62/64
    CategoryConfig(name="KXMLBHR"),  # pnl=$+344.90 win=56/81
    CategoryConfig(name="KXWCMENTION"),  # pnl=$+199.37 win=43/92
    CategoryConfig(name="KXPGATOUR"),  # pnl=$+186.91 win=21/40
    CategoryConfig(name="KXBTCD"),  # pnl=$+182.27 win=44/116
    CategoryConfig(name="KXMLBSPREAD"),  # pnl=$+176.12 win=39/64
    CategoryConfig(name="KXWCAST"),  # pnl=$+149.54 win=22/22
    CategoryConfig(name="KXMENWORLDCUP"),  # pnl=$+118.90 win=16/27
    CategoryConfig(name="KXWCROUND"),  # pnl=$+113.67 win=30/47
    CategoryConfig(name="KXWC1HSPREAD"),  # pnl=$+99.18 win=12/14
    CategoryConfig(name="KXMLBF5SPREAD"),  # pnl=$+95.53 win=20/28
    CategoryConfig(name="KXWCGAME"),  # pnl=$+95.46 win=22/38
    CategoryConfig(name="KXWCSOA"),  # pnl=$+91.79 win=22/31
    CategoryConfig(name="KXNEXTTEAMNBA"),  # pnl=$+90.57 win=16/25
    CategoryConfig(name="KXWCFIRSTGOAL"),  # pnl=$+83.54 win=11/11
    CategoryConfig(name="KXWCMOV"),  # pnl=$+81.37 win=16/19
    CategoryConfig(name="KXWCTCORNERS"),  # pnl=$+74.86 win=15/23
    CategoryConfig(name="KXLIUSAELIMINATION"),  # pnl=$+68.53 win=7/8
    CategoryConfig(name="KXWC2HSPREAD"),  # pnl=$+68.08 win=8/8
    CategoryConfig(name="KXWCSPREAD"),  # pnl=$+64.92 win=12/18
    CategoryConfig(name="KXITFMATCH"),  # pnl=$+59.83 win=19/36
    CategoryConfig(name="KXWCGOALLEADER"),  # pnl=$+57.20 win=10/14
    CategoryConfig(name="KXWC1HBTTS"),  # pnl=$+56.02 win=8/8
    CategoryConfig(name="KXETHD"),  # pnl=$+53.93 win=15/28
    CategoryConfig(name="KXMLBKS"),  # pnl=$+51.71 win=17/31
    CategoryConfig(name="KXWC1H"),  # pnl=$+46.22 win=9/14
    CategoryConfig(name="KXLIUSACOUPLE"),  # pnl=$+45.60 win=6/8
    CategoryConfig(name="KXWCADVANCE"),  # pnl=$+44.69 win=8/25
    CategoryConfig(name="KXBTC"),  # pnl=$+43.76 win=8/13
    CategoryConfig(name="KXWCFURTHESTADVANCING"),  # pnl=$+41.87 win=8/13
    CategoryConfig(name="KXLIUSAELIMINATIONW"),  # pnl=$+40.78 win=14/26
    CategoryConfig(name="KXATP"),  # pnl=$+40.32 win=6/7
    CategoryConfig(name="KXMLBGAME"),  # pnl=$+39.30 win=6/58
    CategoryConfig(name="KXWCSTAGEOFELIM"),  # pnl=$+39.25 win=19/34
    CategoryConfig(name="KXLOVEISLMENTION"),  # pnl=$+39.09 win=5/15
    CategoryConfig(name="KXATPMATCH"),  # pnl=$+37.89 win=19/94
    CategoryConfig(name="KXBTC15M"),  # pnl=$+35.39 win=4/6
    CategoryConfig(name="KXSOLD"),  # pnl=$+35.07 win=7/17
    CategoryConfig(name="KXWC1HTOTAL"),  # pnl=$+33.12 win=10/16
    CategoryConfig(name="KXGTAPREORDERS"),  # pnl=$+32.17 win=5/5
    CategoryConfig(name="KXMLBF5"),  # pnl=$+31.44 win=10/16
    CategoryConfig(name="KXAAAGASM"),  # pnl=$+29.98 win=4/6
    CategoryConfig(name="KXWCFTTS"),  # pnl=$+25.53 win=8/12
    CategoryConfig(name="KXNASCARRACE"),  # pnl=$+23.49 win=3/3
    CategoryConfig(name="KXTRUMPADMINLEAVE"),  # pnl=$+21.90 win=8/9
    CategoryConfig(name="KXWCBTTS"),  # pnl=$+21.08 win=5/8
    CategoryConfig(name="KXWCTEAMTOTAL"),  # pnl=$+20.45 win=4/20
    CategoryConfig(name="KXTEMPNYCH"),  # pnl=$+20.00 win=3/8
    CategoryConfig(name="KXSPACEXCOUNT"),  # pnl=$+19.58 win=2/3
    CategoryConfig(name="KXPGATOP20"),  # pnl=$+18.97 win=7/8
    CategoryConfig(name="KXTRUMPSAYNICKNAME"),  # pnl=$+18.86 win=4/8
    CategoryConfig(name="KXWCTEAMFIRSTGOAL"),  # pnl=$+17.98 win=4/4
    CategoryConfig(name="KXLOVEISLANDUSARANK"),  # pnl=$+17.91 win=5/8
    CategoryConfig(name="KXWC2HBTTS"),  # pnl=$+15.62 win=6/8
    CategoryConfig(name="KXWCTOTAL"),  # pnl=$+15.29 win=12/24
    CategoryConfig(name="KXMLBF5TOTAL"),  # pnl=$+14.51 win=17/44
    CategoryConfig(name="KXLOWTSATX"),  # pnl=$+13.33 win=4/9
    CategoryConfig(name="KXWCMOF"),  # pnl=$+12.46 win=2/4
    CategoryConfig(name="KXXRP15M"),  # pnl=$+11.68 win=2/3
    CategoryConfig(name="KXITFWMATCH"),  # pnl=$+10.70 win=6/17
    CategoryConfig(name="KXAPRPOTUS"),  # pnl=$+10.40 win=2/8
    CategoryConfig(name="KXMLBRFI"),  # pnl=$+9.76 win=1/10
    CategoryConfig(name="KXBNB15M"),  # pnl=$+9.75 win=2/3
    CategoryConfig(name="KXBTCMINMON"),  # pnl=$+9.01 win=3/3
    CategoryConfig(name="KX14AMENDCASE"),  # pnl=$+8.39 win=1/1
    CategoryConfig(name="KXDOGE15M"),  # pnl=$+8.25 win=1/3
    CategoryConfig(name="KXWT20MATCH"),  # pnl=$+8.23 win=2/4
    CategoryConfig(name="KXTESTMATCH"),  # pnl=$+8.16 win=1/2
    CategoryConfig(name="KXRANKLISTSONGSPOTUSA"),  # pnl=$+7.45 win=1/4
    CategoryConfig(name="KXLOWTMIA"),  # pnl=$+5.06 win=7/10
    CategoryConfig(name="KXMADDOWMENTION"),  # pnl=$+4.53 win=2/13
    CategoryConfig(name="KXBTCMAX100"),  # pnl=$+4.40 win=1/1
    CategoryConfig(name="KXWCAWARD"),  # pnl=$+3.71 win=1/4
    CategoryConfig(name="KXMUSKNW"),  # pnl=$+3.59 win=4/5
    CategoryConfig(name="KXBTC2026200"),  # pnl=$+3.59 win=1/1
    CategoryConfig(name="KXHYPE15M"),  # pnl=$+2.27 win=1/3
    CategoryConfig(name="KXSOL15M"),  # pnl=$+2.14 win=1/3
    CategoryConfig(name="KXHIGHPHIL"),  # pnl=$+1.21 win=2/5
    CategoryConfig(name="KXRT"),  # pnl=$+1.18 win=1/9
    CategoryConfig(name="KXHIGHNY"),  # pnl=$+1.12 win=3/6
    CategoryConfig(name="KXBNBD"),  # pnl=$+0.48 win=1/3
    CategoryConfig(name="KXHIGHTMIN"),  # pnl=$+0.32 win=1/7
    CategoryConfig(name="KXHIGHTPHX"),  # pnl=$+0.16 win=1/7
]


TRADING_CONFIG = {
    "capital": 91.66,
    "max_daily_loss": 10.0,
    "stop_loss_threshold": -20.0,
    "max_position_value": 50.0,
}


MONITORING_CONFIG = {
    "log_interval_seconds": 60,
    "health_check_interval": 300,
}
