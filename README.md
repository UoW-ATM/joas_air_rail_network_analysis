# JOAS - Air rail network analysis (short-haul ban)

## Index

1. [About](#about)
    1. [Abstract](#about_abstract)
2. [Setup](#setup)
3. [Data](#data)
    1. [Data availability](#data_availability)
    2. [Brief data description](#data_brief_description)
    3. [Data usage / input / output](#data_usage)
4. [Computation](#cmpt)
   1. [1_generate_schedules_opensky.py](#cmpt_1)
   2. [2_compute_rail_alternatives_flight_schedules.py](#cmpt_2)
   3. [3_process_air_rail_bans.py](#cmpt_3)
   4. [4_compute_plot_results_air_ban.py](#cmpt_4)
   5. [5_fleet_analysis.Rmd](#cmpt_5)
5. [Authorship and License](#license)



## 1. About <a name="about"></a>

Repository with code used to generate analysis and results of the article:

Delgado, L., Trapote-Barreira, C., Montlaur, A., Bolić, T., & Gurtner, G. (2023). _Airlines’ network analysis on an air-rail multimodal system_. Journal of Open Aviation Science, 1(2). https://doi.org/10.59490/joas.2023.7223

### i. Abstract <a name="about_abstract"></a>
This article explores the potential impact of short-haul flight bans in Spain. We build the rail and flight network for the Spanish peninsula, merging openly available ADS-B-based data, for the reconstruction of air schedules and aircraft rotations, and rail operator data, for the modelling of the rail network. We then simulate a ban that would remove flights having a suitable train replacement, i.e., representing a trip shorter than a threshold that we vary continuously up to 15 hours. We study the impact in terms of 1) air route reduction, 2) aircraft utilisation and fleet downsizing for airlines, 3) airport infrastructure relief and rail network requirements, 4) CO<sub>2</sub> emissions, and 5) possible itineraries and travel times for passengers. We find that a threshold of 3 hours (banning all flights with a direct rail alternative faster than three hours) presents some notable advantages in emissions while keeping the aircraft utilisation rate at an adequate level. Interestingly, the passengers would then experience an increase in their itinerary options, with only a moderate increase in their total travelling times.

## 2. Setup <a name="setup"></a>

- Cloning the repository:

```commandline
git clone https://github.com/UoW-ATM/joas_air_rail_network_analysis
```

The code to compute the schedules, rail alternatives and impact of an aviation ban threshold in terms of affected flights, rail usage, potential passenger itineraries, emissions estimations, etc., has been developed in Python (tested on Python 3.8). The analysis of fleet usage is performed in R. 

For Python:
- Install all required packages (code tested with Python 3.8):
```commandline
pip install -r requirements.txt
```


## 3. Data <a name="data"></a>

### i. Data availability <a name="data_availability"></a>
All required data to reproduce the analysis of the article can be downloaded from Zenodo: https://zenodo.org/records/10038841

All data sources used for this article are open source and available from (accessed 25/10/2023) [OpenSky](https://opensky-network.org/data/impala) and [Renfe](https://data.renfe.com/dataset/horarios-de-alta-velocidad-larga-distancia-y-media-distancia). 

In addition, some data has been extracted from open available models and tools such as:
- [EcoPassenger](http://ecopassenger.hafas.de) for rail emissions estimations.
- Flight emissions estimated with models from Montlaur, A., Delgado, L., & Trapote-Barreira, C. (2021). [_Analytical Models for CO<sub>2</sub> Emissions and Travel Time for Short-to-Medium-Haul Flights Considering Available Seats_](https://doi.org/10.3390/su131810401). Sustainability 13.18 (2021) and from some specific flights using EUROCONTROL's [IMPACT](www.eurocontrol.int/platform/integrated-aircraft-noise-and-emissions-modelling-platform) model.

The authors have also compiled additional datasets, such as airport data (e.g. location), min and max number of seats per aircraft type, manual flight rotations fixes, etc. All these additional datasets are also available in the Zenodo record.

### ii. Brief data description <a name="data_brief_description"></a>
Brief description of data input:
- From [OpenSky](https://opensky-network.org/data/impala):
  - flight information from OpenSky's flights_data4 table for the first week of May 2023.
  - aircraft database (registration, type, etc.) as a function of transponder's icao24 id.
- From [Renfe](https://data.renfe.com/dataset/horarios-de-alta-velocidad-larga-distancia-y-media-distancia):
  - mid and long-distance train dataset (GTFS): routes, stops, stops times, trips, etc. for 2023.
- Pre-computed from other sources (by authors):
  - rail emissions estimated using [EcoPassenger](http://ecopassenger.hafas.de).
  - flight emissions per route estimated based on the model from Montlaur, A., Delgado, L., & Trapote-Barreira, C. (2021). [_Analytical Models for CO<sub>2</sub> Emissions and Travel Time for Short-to-Medium-Haul Flights Considering Available Seats_](https://doi.org/10.3390/su131810401). Sustainability 13.18 (2021) and from some specific flights using EUROCONTROL's [IMPACT](https://www.eurocontrol.int/platform/integrated-aircraft-noise-and-emissions-modelling-platform) model.
- Other (collected/computed by authors):
  - airport static information (coordinates).
  - manually modified airport codes with the list of airport codes swapped as erroneous departure or arrival sourced from OpenSky (flights_data4 table).
  - aircraft rotations with further corrections and identification of the first and last airport visited daily. Compiled by own development and with additional data from [FlightRadar24](https://www.flightradar24.com/).
  - min and max number of seats for aircraft type from airlines' websites and other sources.
  - aircraft type for some transponder's icao24 identifiers missing from OpenSky, compiled from [FlightRadar24](https://www.flightradar24.com/).

### iii. Data usage / input / output <a name="data_usage"></a>

- The scripts read the data from a folder called _data_ and produce output in a folder called _output_. These folders are expected to be in the same folder as the scripts by default.
- Paths can be modified directly in the code.
- The outcome of some scripts becomes part of the input to others, so the _downstream_ scripts will look for the data in the _output_ folder.


- By default:
  - Input data stored in _data_ folder
    - flights_data4 data stored in _flights_data4_ sub-folder following a data lake architecture (i.e., flights_data4/year=2023/month=05/)
    - renfe data stored in _renfe_ sub-folder
    - pre-computed data stored in _data_computed_ sub-folder
    - other datasets directly in _data_ folder
     
  - Output folder will be automatically generated (_output_)
    - Computed results from flight data (e.g. schedules) will be stored in _air_ sub-folder
    - Computed results from rail data (e.g. train services to replace flights) will be stored in _rail_ sub-folder
    - Computed results for multimodal analysis (ban analysis) will be stored in _multi_ sub-folder
    - All figures generated will be stored in _figs_ sub-folder
    - Logs are stored in _log_ sub-folder



## 4. Computation <a name="cmpt"></a>
The scripts are numbered for simplification of the computation and analysis.
All scripts have a set of parameters that can be tuned (e.g. path to input/output files/folders).

The code is in Python, except for the scripts required for analysing the fleet usage, which are provided as R notebooks.

### i. 1_generate_schedules_opensky.py <a name="cmpt_1"></a>
This code estimates the flight schedules from the OpenSky historic flight_data4 table.
The output (estimated schedules) will be stored in the _output_ folder (/output/air/)

### ii. 2_compute_rail_alternatives_flight_schedules.py <a name="cmpt_2"></a>
This code computes the possible rail alternatives for the flights using the Renfe GTFS dataset.
The output (possible rail alternatives) is stored in the _output_ folder (/output/rail/)

### iii. 3_process_air_rail_bans.py <a name="cmpt_3"></a>
This code is the main one computing the impact of the rail bans.
It computes the possible itineraries as a function of the ban replacing potential flight trips for multimodal or only rail alternatives.
For each day of analysis and ban threshold, it computes the flights and rails used and the possible itineraries to be used by the passengers to fulfil (as much as possible) the initial flight connectivity of the network.

### iv. 4_compute_plot_results_air_ban.py <a name="cmpt_4"></a>
This code uses the flight and rail used as a function of the threshold ban (and dah) and the possible itineraries
to analyse the system in different mobility performances and produce the required plots (emissions, number of routes used, number of potential itineraries (flight, flight-flight, rail-flight, flight-rail and rail)).
Results are mainly stored in the _output_ folder as figures (/output/figs/) with some analysis as csv files (/output/multi/).

This code also produces (based on the previous computations) the following figures:
- maps of flight and rail replacement as a function of the ban threshold (Figure 2 in the article) (output in output/figs/img_map_replacement/), 
- average number of routes per day per airline as a function of the ban threshold (Figure 3 in the article), 
- demand analysis at Madrid Barajas airport (LEMD) (Figure 7 in the article),
- daily average emissions shifted from air to rail as a function of the ban threshold (Figure 8 in the article),
- average number of possible itineraries within Peninsular Spain per trip type (flight, flight-flight, rail, rail-flight and flight-rail) as a function of the ban threshold (Figure 9 in the article),
- mean time of possible itineraries within Peninsular Spain as a function of trip type (Figure 10 in the article)

### v. 5_fleet_analysis.Rmd <a name="cmpt_5"></a>
R Notebook code to analyse the fleet usage with the short-haul ban consideration. 

It generates the figures for:
- the evolution of airline's utilisation factor as a function of ban threshold (Figure 4 in the article),
- the variation of ground time as a function of the ban threshold (Figure 5 in the article),
- the fleet size variation as a function of the ban threshold (Figure 6 in the article)

In this case, figures are not directly stored but displayed in the notebook.

## 5. Authorship and License <a name="license"></a>
Luis Delgado and Cesar Trapote-Barreira have written all the code. Data preparation, analysis and overall research were done with Adeline Montlaur, Tatjana Bolić and Gérald Gurtner.

The code is released under the GPL v3 licence.

Copyright 2023 Luis Delgado, Cesar Trapote-Barreira.

All subsequent copyright belongs to the respective contributors.
