import pandas as pd
import numpy as np
from sklearn.svm import SVR

SEASON_MAP = {
    "2007/08": 1, "2009": 2, "2009/10": 3, "2011": 4, "2012": 5, "2013": 6, "2014": 7, "2015": 8, "2016": 9, "2017": 10,
    "2018": 11, "2019": 12, "2020/21": 13, "2021": 14, "2022": 15, "2023": 16, "2024": 17, "2025": 18, "2026": 19,
}

VENUE_NORM = {
    "ma chidambaram stadium": "MA Chidambaram Stadium","ma chidambaram stadium, chepauk": "MA Chidambaram Stadium", "ma chidambaram stadium, chepauk, chennai": "MA Chidambaram Stadium",
    "m chinnaswamy stadium": "M Chinnaswamy Stadium","m.chinnaswamy stadium": "M Chinnaswamy Stadium","m chinnaswamy stadium, bengaluru": "M Chinnaswamy Stadium",
    "wankhede stadium": "Wankhede Stadium","wankhede stadium, mumbai": "Wankhede Stadium",
    "narendra modi stadium": "Narendra Modi Stadium, Ahmedabad","narendra modi stadium, ahmedabad": "Narendra Modi Stadium, Ahmedabad",
    "sardar patel stadium, motera": "Narendra Modi Stadium, Ahmedabad","sardar patel stadium": "Narendra Modi Stadium, Ahmedabad","motera stadium": "Narendra Modi Stadium, Ahmedabad",
    "eden gardens": "Eden Gardens","eden gardens, kolkata": "Eden Gardens",
    "sawai mansingh stadium": "Sawai Mansingh Stadium","sawai mansingh stadium, jaipur": "Sawai Mansingh Stadium",
    "barsapara cricket stadium": "Barsapara Cricket Stadium, Guwahati","barsapara stadium, guwahati": "Barsapara Cricket Stadium, Guwahati","barsapara cricket stadium, guwahati": "Barsapara Cricket Stadium, Guwahati",
    "arun jaitley stadium": "Arun Jaitley Stadium","arun jaitley stadium, delhi": "Arun Jaitley Stadium",
    "feroz shah kotla": "Arun Jaitley Stadium","feroz shah kotla ground": "Arun Jaitley Stadium",
    "dr. y.s. rajasekhara reddy aca-vdca cricket stadium": "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium","dr. y.s. rajasekhara reddy aca-vdca cricket stadium, visakhapatnam": "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium",
    "punjab cricket association is bindra stadium": "Punjab Cricket Association IS Bindra Stadium","punjab cricket association is bindra stadium, mohali": "Punjab Cricket Association IS Bindra Stadium","punjab cricket association is bindra stadium, mohali, chandigarh": "Punjab Cricket Association IS Bindra Stadium","punjab cricket association stadium, mohali": "Punjab Cricket Association IS Bindra Stadium",
    "himachal pradesh cricket association stadium": "Himachal Pradesh Cricket Association Stadium", "himachal pradesh cricket association stadium, dharamsala": "Himachal Pradesh Cricket Association Stadium",
    "rajiv gandhi international stadium": "Rajiv Gandhi International Stadium","rajiv gandhi international stadium, uppal": "Rajiv Gandhi International Stadium","rajiv gandhi international stadium, uppal, hyderabad": "Rajiv Gandhi International Stadium","rajiv gandhi international cricket stadium": "Rajiv Gandhi International Stadium",
    "bharat ratna shri atal bihari vajpayee ekana cricket stadium": "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium","bharat ratna shri atal bihari vajpayee ekana cricket stadium, lucknow": "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium",
    "brabourne stadium": "Brabourne Stadium","brabourne stadium, mumbai": "Brabourne Stadium",
    "dr dy patil sports academy": "Dr DY Patil Sports Academy","dr dy patil sports academy, mumbai": "Dr DY Patil Sports Academy",
    "maharashtra cricket association stadium": "Maharashtra Cricket Association Stadium","maharashtra cricket association stadium, pune": "Maharashtra Cricket Association Stadium",
    "maharaja yadavindra singh international cricket stadium, mullanpur": "Maharaja Yadavindra Singh International Cricket Stadium","maharaja yadavindra singh international cricket stadium, new chandigarh": "Maharaja Yadavindra Singh International Cricket Stadium",
}

def _norm_venue(name: str) -> str:
    key = str(name).strip().lower()
    return VENUE_NORM.get(key, str(name).strip())

ACTIVE_TEAMS = {
    "Chennai Super Kings", "Delhi Capitals", "Gujarat Titans", "Kolkata Knight Riders",
    "Lucknow Super Giants", "Mumbai Indians", "Punjab Kings", "Rajasthan Royals",
    "Royal Challengers Bengaluru", "Sunrisers Hyderabad",
}

TEAM_NORM = {
    "royal challengers bangalore":  "Royal Challengers Bengaluru","royal challengers bengaluru":  "Royal Challengers Bengaluru",
    "delhi daredevils": "Delhi Capitals","delhi capitals": "Delhi Capitals",
    "kings xi punjab": "Punjab Kings","punjab kings": "Punjab Kings",
    "rising pune supergiant": "Rising Pune Supergiants","rising pune supergiants": "Rising Pune Supergiants",
    "sunrisers hyderabad": "Sunrisers Hyderabad",
}

def _norm_team(name: str) -> str:
    key = str(name).strip().lower()
    canonical = TEAM_NORM.get(key, str(name).strip())
    return canonical if canonical in ACTIVE_TEAMS else "Other"


def _parse_wickets(player_id_str) -> int:
    if pd.isna(player_id_str) or str(player_id_str).strip() == "":
        return 0
    ids = [p.strip() for p in str(player_id_str).split(",") if p.strip()]
    return int(np.clip(len(ids) - 2, 0, 10))


def _ceil_to_5(x: float) -> int:
    x = int(x)
    return x if x % 5 == 0 else int(np.ceil(x / 5.0) * 5)


class MyModel:

    def __init__(self):
        self.model        = None
        self.global_mean  = 50
        self.venue_map    = {}
        self.team_columns = []

        self._v_wkt_mean      = {}
        self._v_inn_mean      = {}
        self._v_six_mean      = {}
        self._v_six_season_mean = {}
        self._v_season_mean   = {}
        self._h2h_venue_mean  = {}
        self._bat_venue_mean  = {}   # venue x bat_team — h2h fallback level 2
        self._bat_season_mean = {}   # bat_team x season — bat_form fallback level 2
        self._bat_form        = {}
        self._v_trend         = {}
        self._global_v_wkt    = 50.0
        self._global_v_inn    = 50.0
        self._global_six_mean = 3.0

    def _build_features(self, agg: pd.DataFrame) -> pd.DataFrame:
        gm      = self._global_v_wkt   # global run mean — last resort fallback
        last_s  = max(self._v_season_mean.keys(), key=lambda x: x[1], default=(0, 0))[1]

        # v_wkt_mean: venue x wickets → venue x inning → global
        agg["v_wkt_mean"] = [
            self._v_wkt_mean.get((v, w),
            self._v_inn_mean.get((v, i),
            gm))
            for v, w, i in zip(agg["v_enc"], agg["wickets"], agg["inning"])
        ]

        # v_inn_mean: venue x inning → global
        agg["v_inn_mean"] = [
            self._v_inn_mean.get((v, i), gm)
            for v, i in zip(agg["v_enc"], agg["inning"])
        ]

        # v_six_season_mean: venue x season → venue all-time → global six mean
        agg["v_six_mean"] = [
            self._v_six_mean.get(v, self._global_six_mean)
            for v in agg["v_enc"]
        ]
        agg["v_six_season_mean"] = [
            self._v_six_season_mean.get((v, s),
            self._v_six_mean.get(v, self._global_six_mean))
            for v, s in zip(agg["v_enc"], agg["season"])
        ]

        # v_season_mean: venue x season → venue x last known season → venue x inning → global
        agg["v_season_mean"] = [
            self._v_season_mean.get((v, s),
            self._v_season_mean.get((v, last_s),
            self._v_inn_mean.get((v, i),
            gm)))
            for v, s, i in zip(agg["v_enc"], agg["season"], agg["inning"])
        ]

        # h2h_venue_mean: venue x bat x bowl → venue x bat → venue x inning → global
        agg["h2h_venue_mean"] = [
            self._h2h_venue_mean.get((v, bt, bw),
            self._bat_venue_mean.get((v, bt),
            self._v_inn_mean.get((v, i),
            gm)))
            for v, bt, bw, i in zip(agg["v_enc"], agg["batting_team"], agg["bowling_team"], agg["inning"])
        ]

        # bat_form_3: team rolling form → team last-season avg → venue x inning → global
        agg["bat_form_3"] = [
            self._bat_form.get(bt,
            self._bat_season_mean.get((bt, last_s),
            self._v_inn_mean.get((v, i),
            gm)))
            for bt, v, i in zip(agg["batting_team"], agg["v_enc"], agg["inning"])
        ]

        agg["v_trend"] = [
            self._v_trend.get(v, 0.0) for v in agg["v_enc"]
        ]
        return agg

    def fit(self, deliveries_df, players_df=None, matches_df=None):

        df = deliveries_df.copy()
        df["over"]   = pd.to_numeric(df["over"],   errors="coerce")
        df["inning"] = pd.to_numeric(df["inning"], errors="coerce").astype("int32")

        run_cols = ["batsman_runs", "extras", "isWide", "isNoBall", "Byes", "LegByes"]
        run_cols = [c for c in run_cols if c in df.columns]
        df[run_cols] = df[run_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

        df = df[df["over"] < 6].copy()
        df["ball_runs"] = df[run_cols].sum(axis=1)
        df["is_wicket"] = df.get("player_dismissed", pd.Series()).notna().astype(int)
        df["is_six"] = (
            (df["batsman_runs"] == 6) &
            (df.get("isWide",   pd.Series(0, index=df.index)).fillna(0) == 0) &
            (df.get("isNoBall", pd.Series(0, index=df.index)).fillna(0) == 0)
        ).astype(int)

        agg = (
            df.groupby(["matchId", "inning"])
            .agg(
                runs=("ball_runs", "sum"),
                wickets=("is_wicket", "sum"),
                sixes=("is_six", "sum"),
                batting_team=("batting_team", "first"),
                bowling_team=("bowling_team", "first"),
            )
            .reset_index()
        )

        agg["batting_team"] = agg["batting_team"].apply(_norm_team)
        agg["bowling_team"] = agg["bowling_team"].apply(_norm_team)

        meta_cols = ["matchId", "venue", "season"]
        for opt in ["toss_winner", "toss_decision"]:
            if opt in matches_df.columns:
                meta_cols.append(opt)

        meta = matches_df[meta_cols].drop_duplicates("matchId").copy()
        meta["venue"] = meta["venue"].astype(str).apply(_norm_venue)
        agg = agg.merge(meta, on="matchId", how="left")

        agg["venue"]  = agg["venue"].fillna("Unknown")
        agg["season"] = (
            agg["season"].fillna("Unknown").astype(str)
            .map(lambda x: SEASON_MAP.get(x.strip(), 0)).astype("int16")
        )

        if "toss_winner" in agg.columns and "toss_decision" in agg.columns:
            agg["toss_bat"] = (
                (agg["batting_team"] == agg["toss_winner"]) &
                (agg["toss_decision"].str.lower().str.strip() == "bat")
            ).astype(int)
        else:
            agg["toss_bat"] = 0

        self.venue_map = {v: i for i, v in enumerate(sorted(agg["venue"].unique()))}
        agg["v_enc"] = agg["venue"].map(self.venue_map).fillna(-1).astype(int)

        gm = float(agg["runs"].mean())
        self._global_v_wkt    = gm
        self._global_v_inn    = gm
        self._global_six_mean = float(agg["sixes"].mean())
        self.global_mean      = int(round(gm))

        self._v_wkt_mean        = agg.groupby(["v_enc", "wickets"])["runs"].mean().to_dict()
        self._v_inn_mean        = agg.groupby(["v_enc", "inning"]) ["runs"].mean().to_dict()
        self._v_six_mean        = agg.groupby("v_enc")["sixes"].mean().to_dict()
        self._v_six_season_mean = agg.groupby(["v_enc", "season"])["sixes"].mean().to_dict()
        self._v_season_mean     = agg.groupby(["v_enc", "season"])["runs"].mean().to_dict()
        self._h2h_venue_mean    = agg.groupby(["v_enc", "batting_team", "bowling_team"])["runs"].mean().to_dict()
        self._bat_venue_mean    = agg.groupby(["v_enc", "batting_team"])["runs"].mean().to_dict()
        self._bat_season_mean   = agg.groupby(["batting_team", "season"])["runs"].mean().to_dict()

        agg_sorted = agg.sort_values(["season", "matchId"])
        bat_form_series = (
            agg_sorted.groupby("batting_team")["runs"]
            .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        )
        form_df = agg_sorted.copy()
        form_df["bat_form_3"] = bat_form_series.reindex(agg_sorted.index).fillna(gm)
        self._bat_form = form_df.groupby("batting_team")["bat_form_3"].last().to_dict()

        vsm = agg.groupby(["v_enc", "season"])["runs"].mean().reset_index()
        def _slope(grp):
            if len(grp) < 2:
                return 0.0
            x = grp["season"].values.astype(float)
            return float(np.polyfit(x, grp["runs"].values, 1)[0])
        self._v_trend = vsm.groupby("v_enc").apply(_slope).to_dict()

        agg = self._build_features(agg)

        bat_ohe  = pd.get_dummies(agg["batting_team"],  prefix="bat")
        bowl_ohe = pd.get_dummies(agg["bowling_team"],  prefix="bowl")
        self.team_columns = list(bat_ohe.columns) + list(bowl_ohe.columns)

        base_features = [
            "v_enc", "inning", "wickets", "season",
            "v_wkt_mean", "v_inn_mean",
            "v_six_mean", "v_six_season_mean",
            "h2h_venue_mean",   
            "v_season_mean",    
            "bat_form_3",       
            "v_trend",          
            "toss_bat",        
        ]
        X = pd.concat([
            agg[base_features].reset_index(drop=True),
            bat_ohe.reset_index(drop=True),
            bowl_ohe.reset_index(drop=True),
        ], axis=1)
        y = agg["runs"].values.astype("float32")

        self.model = SVR(kernel='rbf', gamma='scale', C=113.084, epsilon=3.418)
        self.model.fit(X.values.astype("float32"), y)
        self._feature_names = list(X.columns)

        return self

    def predict(self, test_df):

        records = []

        for i, row in test_df.iterrows():
            inning       = int(row.get("innings", 1))
            venue        = _norm_venue(str(row.get("venue", "Unknown")))
            season       = int(SEASON_MAP.get(str(row.get("season", "2026")), 19))
            wickets      = _parse_wickets(row.get("Batsman's Player Id", ""))
            v_enc        = self.venue_map.get(venue, -1)
            batting_team = _norm_team(str(row.get("batting_team", "Unknown")))
            bowling_team = _norm_team(str(row.get("bowling_team", "Unknown")))

            toss_winner   = str(row.get("toss_winner",   ""))
            toss_decision = str(row.get("toss_decision", "")).strip().lower()
            toss_bat = int(batting_team == toss_winner and toss_decision == "bat")
            row_df = pd.DataFrame([{
                "v_enc":        v_enc,
                "inning":       inning,
                "wickets":      wickets,
                "season":       season,
                "batting_team": batting_team,
                "bowling_team": bowling_team,
                "toss_bat":     toss_bat,
            }])
            row_df = self._build_features(row_df)

            base_features = [
                "v_enc", "inning", "wickets", "season",
                "v_wkt_mean", "v_inn_mean",
                "v_six_mean", "v_six_season_mean",
                "v_season_mean", "h2h_venue_mean", "bat_form_3", "v_trend", "toss_bat",
            ]
            base = row_df[base_features].reset_index(drop=True)

            team_row = pd.DataFrame([[0] * len(self.team_columns)], columns=self.team_columns)
            bat_col  = f"bat_{batting_team}"
            bowl_col = f"bowl_{bowling_team}"
            if bat_col  in team_row.columns: team_row[bat_col]  = 1
            if bowl_col in team_row.columns: team_row[bowl_col] = 1

            x = pd.concat([base, team_row], axis=1).values.astype("float32")

            if self.model:
                raw  = float(self.model.predict(x)[0])
                pred = _ceil_to_5(raw)
            else:
                pred = self.global_mean

            records.append({"id": row.get("id", i), "predicted_score": pred})

        return pd.DataFrame(records)