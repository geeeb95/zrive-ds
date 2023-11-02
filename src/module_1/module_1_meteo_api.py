import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
from ratelimit import limits, sleep_and_retry


COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}

VARIABLES = "temperature_2m_mean,precipitation_sum,soil_moisture_0_to_10cm_mean"


def get_data_meteo_api(lat: float, long: float, variable: str):
    # For the API call I only parametrized the coordinates as
    # they are the only parameters that will vary in this case
    API_URL_0 = f"https://climate-api.open-meteo.com/v1/climate?latitude={lat}&longitude={long}&start_date=1950-01-01&end_date=2050-12-31&models=CMCC_CM2_VHR4,FGOALS_f3_H,HiRAM_SIT_HR,MRI_AGCM3_2_S,EC_Earth3P_HR,MPI_ESM1_2_XR,NICAM16_8S&daily="
    API_URL = API_URL_0 + variable
    return pd.DataFrame(json.loads(api_request_manager(API_URL)))


# This function controls that the API call limit is not exceeded
# and also raises an error if the status is not 200
@sleep_and_retry
@limits(calls=10000, period=timedelta(days=1).total_seconds())
def api_request_manager(URL):
    response = requests.get(URL)
    response.raise_for_status()
    return response.content.decode("utf-8")


def calculate_mean_std(data: pd.DataFrame):
    # Daily values for different models are put into a matrix to calculate mean and std
    matrix = [np.array(data.iloc[0], dtype=np.float64)]
    for n, models in enumerate(data.iloc[1:]):
        array = np.array(data.iloc[n+1], dtype=np.float64)
        # The following line raises an error with mypy that I don't know how to fix,
        # I believe it has to do with treating the array as a list. There is also
        array[array is None] = np.nan
        matrix = np.append(matrix, [array], axis=0)

    return np.nanmean(matrix, axis=0), np.nanstd(matrix, axis=0)


def plot_data(data: pd.DataFrame):
    # Grouping by year
    data["Time"] = pd.to_datetime(data["Time"])
    data["Years"] = data["Time"].dt.year
    data_group = data.groupby(pd.Grouper(key="Time", freq="Y")).mean()
    # Figure parameters
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    units = ["ºC", "mm", "${m^3}/{m^3}$"]
    colors = ["b", "g", "r"]
    # Plotting all the data iteratively
    for i, var in enumerate(VARIABLES.split(",")):
        axs[i].title.set_text(var)
        axs[i].set_xlabel("Years")
        axs[i].set_ylabel(units[i])
        for n, city in enumerate(COORDINATES.keys()):
            for col in data_group.columns:
                if col.startswith(city) and col.endswith(var):
                    if col.startswith(city + "_mean_"):
                        axs[i].plot(data_group["Years"], data_group[col],
                                    color=colors[n], label=city + "_mean")
                        mean = col
                    else:
                        up = data_group[mean]+data_group[col]
                        down = data_group[mean]-data_group[col]
                        axs[i].fill_between(data_group["Years"], up, down, alpha=0.4,
                                            color=colors[n], label=city + "_std")
    # Legend and subplot layout adjustments
    fig.tight_layout()
    handles, labels = axs[i].get_legend_handles_labels()
    fig.subplots_adjust(bottom=0.08)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels))
    plt.savefig("src/module_1/fig.png")


def main():
    # The variables are requested iteratively and their mean and
    # std calculated, totalling 3*3 = 9 API calls
    # everything is stored in a dictionary and then
    # converted into a dataframe in order to have properly named columns
    data_dict = {}
    for city in COORDINATES.keys():
        for var in VARIABLES.split(","):
            variable = get_data_meteo_api(COORDINATES[city]["latitude"],
                                          COORDINATES[city]["longitude"], var)["daily"]
            (mean, std) = calculate_mean_std(variable[1:])
            data_dict.update({city+"_mean_"+var: mean})
            data_dict.update({city+"_std_"+var: std})
    # Another key is added for the year
    data_dict.update({"Time": variable.iloc[0]})
    data = pd.DataFrame(data_dict)
    plot_data(data)


if __name__ == "__main__":
    main()
