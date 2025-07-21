import os
import matplotlib.pyplot as plt
import numpy as np
import time
import tensorflow as tf
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'              # hide noisy INFO

tf.keras.backend.set_floatx('float32')                # global policy

print('Start loading data')

time0 = time.time()

out_dir = Path("data")

model = tf.keras.models.load_model(out_dir / 'ANN_model.keras', compile=False)

# --- individual arrays
energies_mid   = np.loadtxt(out_dir / "energies_mid.txt")
lower_energy   = np.loadtxt(out_dir / "lower_energy.txt")
upper_energy    = np.loadtxt(out_dir / "upper_energy.txt")
table_all_cs = np.loadtxt(out_dir / "table_all_cs.txt")

print('Spend time for loading data : ', time.time() - time0, "s")


tabel_non_inf = table_all_cs.copy()

tabel_non_inf[tabel_non_inf <= 1.0e-13] = np.inf
no_zeros = 1. / tabel_non_inf
no_zeros = no_zeros[no_zeros > 0.]
nb_no_zeros = np.shape(no_zeros)[0]

nb_sub = 1


react_rate = np.loadtxt('reaction_rates.txt')
factor = np.sum(react_rate)
react_rate /= factor

flux_g = react_rate.reshape(len(react_rate), 1) / tabel_non_inf
flux_g_sum = flux_g.reshape(20, int(172 / nb_sub), nb_sub).sum(axis=2)
copied_flux_g = flux_g_sum.copy()
copied_flux_g[copied_flux_g <= 0] = np.inf
flux_g_min = copied_flux_g.min(axis=0)
our_test = flux_g.reshape(flux_g.shape[0] * flux_g.shape[1])
our_test = our_test[our_test > 0.]
our_test = 1. / our_test
our_test = our_test.reshape(1, len(our_test))


time1 = time.time()

ANN_result = model.predict(our_test)

print('Spend time on spectrum calculation : ', time.time() - time1, "s")


ref_spectrum_by_energy = []
ANN_result_by_energy = []
ANN_result = ANN_result.reshape(172)
ANN_result *= flux_g_min
ANN_result *= factor


for i in range(len(energies_mid)):
    ANN_result_by_energy.append(ANN_result[i] / (upper_energy[i] - lower_energy[i]))

fig, ax = plt.subplots()
plt.loglog(energies_mid, ANN_result_by_energy, 'b', label='predicted')
aa, bb = 12, 9
fig.set_size_inches(aa, bb)
fig.tight_layout()
plt.savefig('results_spectrum_.pdf')
plt.show()


# time2 = time.time()
#
# nb_itt = 20
#
# for i in range(nb_itt):
#     result = model.predict(our_test)
#
# print('Average spend time on spectrum calculation : ', (time.time() - time2) / nb_itt, "s")


print('Reproduced reaction rates: ', np.matmul(table_all_cs, ANN_result))

