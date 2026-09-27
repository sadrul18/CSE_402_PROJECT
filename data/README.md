# French RTC solar-cell dataset

`rtc_france.csv` contains the 26 measured I-V points from Table S1 of the
supplementary material for:

Cheng Qin, Jianing Li, Chen Yang, Bin Ai, and Yecheng Zhou, "Comparative Study
of Parameter Extraction from a Solar Cell or a Photovoltaic Module by Combining
Metaheuristic Algorithms with Different Simulation Current Calculation Methods,"
*Energies*, vol. 17, no. 10, 2284, 2024.
https://doi.org/10.3390/en17102284

- Dataset: French RTC solar cell
- Temperature: 33 degrees C (306.15 K)
- Irradiance: 1000 W/m^2
- Cells in series (`ns`): 1
- Original columns: experimental voltage `Vexp` in V and current `Iexp` in A
- Preprocessing: selected the measured `Vexp` and `Iexp` columns; preserved all
  26 points, their published precision, signs, and order
- Source file: `energies-2959298-supplementary.pdf`, Table S1
- Reuse terms: the article is published under CC BY 4.0

The calculated-current and power columns in Table S1 are results from the paper,
so they are not included as input measurements.
