import polars as pl


from checks.categorical import categorical_checks
from checks.general import general_checks
from checks.numerical import numerical_checks


def main():
    parquet_path = "./Combined_Flights_2022.parquet"
    cat_cols = [
        "Year",
        "Quarter",
        "Month",
        "DayofMonth",
        "DayOfWeek",
        "Operating_Airline",
        "OriginAirportSeqID",
        "OriginCityMarketID",
        "OriginCityName",
        "OriginState",
        "OriginStateFips",
        "OriginStateName",
        "OriginWac",
        "DestAirportID",
        "DestAirportSeqID",
        "DestCityMarketID",
        "Dest",
        "DestCityName",
        "DestState",
        "DestStateFips",
        "DestStateName",
        "DestWac",
    ]
    num_cols = ["CRSElapsedTime", "ActualElapsedTime", "AirTime", "Distance"]
    df = pl.read_parquet(parquet_path)
    general_checks(df)
    if cat_cols:
        categorical_checks(df, cat_cols=cat_cols)
    if num_cols:
        numerical_checks(df, num_cols=num_cols)


if __name__ == "__main__":
    main()
