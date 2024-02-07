import pickle
import matplotlib.pyplot as plt
import numpy as np

"""
Once the rail alternatives are computed and saved in dict_direct_rail_purged.pickle, this code can plot
the number of services available between airports by rail as a function of the day. The code also shows
how to compute the number of services between a given o-d pair for a given day, and the number of services overall
and per day.
"""

output_dict_direct_rail_purged = "./output/rail/dict_direct_rail_purged.pickle"
fig_name = "./output/figs/number_options_rail.png"
day_options = '20230501'

with open(output_dict_direct_rail_purged, "rb") as f:
    dict_direct_rail_purged = pickle.load(f)

# Plots options rail
number_options_rail_day = {}
for k, v in dict_direct_rail_purged.items():
    number_options_rail_day[k] = {}
    for od, r in v.items():
        options = len(r.keys())
        number_options_rail_day[k].update({od: options})

# Number of options in a given day between o-d pairs
print("Options between airports for day: "+day_options)
for od, options in number_options_rail_day[day_options].items():
    print(od, ":", options)

# All od, all days rail
all_od_rail = set()
for k, v in number_options_rail_day.items():
    all_od_rail = all_od_rail.union(set(v.keys()))


list_all = []
days = []
list_order_od = [('LEBL', 'LEMD'), ('LEZL', 'LEMD'), ('LEBL', 'LEZL')]  # Add as many o-d pairs as wanted in the plot
# list_order_od = list(all_od_rail)
for k, v in number_options_rail_day.items():
    days += [k]
    list_k = []
    for od in list_order_od:
        if od in v.keys():
            list_k += [v[od]]
        else:
            # print(od)
            list_k += [0]

    list_all += [list_k]

values = np.array(list_all)

# Plotting
fig, ax = plt.subplots(figsize=(18, 8))

bar_width = 0.1
bar_positions = np.arange(len(list_order_od))

for i, day in enumerate(days):
    ax.bar(bar_positions + i * bar_width, values[i], bar_width, label=day)

# Set labels and title
ax.set_xlabel('O-D pairs')
ax.set_ylabel('Rail services')
ax.set_title('Number options between o-d pair train per day')
ax.set_xticks(bar_positions + bar_width * len(days) / 2)
ax.set_xticklabels(list_order_od, rotation=90)
ax.legend(title='Days')

# Show the plot
plt.savefig(fig_name, transparent=False, facecolor='white', bbox_inches='tight')


# Total number of services
day = '20230507'
n_services = 0
for k, v in number_options_rail_day[day].items():
    n_services += v

print("total number of services on day", day, n_services)

n_services_total = 0
for d, sd in number_options_rail_day.items():
    for k, v in sd.items():
        n_services_total += v

print("Average number services per day", n_services_total/len(number_options_rail_day))
