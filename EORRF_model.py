import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np

# importo dataset
fileUU = "c:/Users/Carlo/Desktop/Tesi/IS_LPMOR/IS_GMS-UU.csv"
fileUD = "c:/Users/Carlo/Desktop/Tesi/IS_LPMOR/IS_GMS-UD.csv"
fileDU = "c:/Users/Carlo/Desktop/Tesi/IS_LPMOR/IS_GMS-DU.csv"
fileDD = "c:/Users/Carlo/Desktop/Tesi/IS_LPMOR/IS_GMS-DD.csv"
fileIT1 = "c:/Users/Carlo/Desktop/Tesi/IS_LPMOR/IS_ORL-IT1.csv"
fileIT2 = "c:/Users/Carlo/Desktop/Tesi/IS_LPMOR/IS_ORL-IT2.csv"
fileIT3 = "c:/Users/Carlo/Desktop/Tesi/IS_LPMOR/IS_ORL-IT3.csv"
fileIT4 = "c:/Users/Carlo/Desktop/Tesi/IS_LPMOR/IS_ORL-IT4.csv"
fileIT5 = "c:/Users/Carlo/Desktop/Tesi/IS_LPMOR/IS_ORL-IT5.csv"
fileIT6 = "c:/Users/Carlo/Desktop/Tesi/IS_LPMOR/IS_ORL-IT6.csv"
fileIT7 = "c:/Users/Carlo/Desktop/Tesi/IS_LPMOR/IS_ORL-IT7.csv"
fileIT8 = "c:/Users/Carlo/Desktop/Tesi/IS_LPMOR/IS_ORL-IT8.csv"

dataset = pd.read_csv(fileIT5, header=None, skiprows=3)

# Number of Stocks and Number of Periods
[N, T] = dataset.shape
[N_assets, T_scenarios] = [N - 1, int((T - 1)/2)]  # QUI MODIFICO, voglio 100x52

# P_t is the probability of each scenario, considering equally probable
p_t = pd.Series(np.ones(T_scenarios)/T_scenarios)
Benchmark = dataset.iloc[-1, np.arange(T_scenarios)]

# pulisco dataframe rimuovendo ultima riga e ultima colonna
df_test = dataset.iloc[np.arange(N_assets), np.arange(T_scenarios)].values
df = pd.DataFrame(df_test)
# Raccolta dati
alfa = 0.003            # incremento sull'obiettivo da raggiungere
mu_j = df.mean(axis=1)   # Costrusico vettore con le medie degli assets indipendentemente dallo scenario
mu_I = Benchmark.mean()  # Media Benchmark
mu_alfa = mu_I+alfa      # obiettivo da raggiungere
M = 1000000              # Uso della grande M
K = 10                   # Massima cardinalità
epsilon = 0.1
delta = 0.3
deltavector = np.ones(N_assets)*delta

# Create an empty model
m = gp.Model(name="Omega Ratio Model")

# Add a variable x_j, v0, v
variable_x = pd.Series(m.addVars(N_assets, name="x_j", lb=0.0))
variable_d = pd.Series(m.addVars(T_scenarios, name="d_t", vtype=GRB.CONTINUOUS, lb=0.0))
variable_utilde = pd.Series(m.addVars(T_scenarios, vtype=GRB.CONTINUOUS, name="u_t", lb=0.0, ub=M))
variable_u = pd.Series(m.addVars(T_scenarios, vtype=GRB.BINARY, name="u_t"))
variable_z = pd.Series(m.addVars(N_assets, vtype=GRB.BINARY, name="z_i"))

somma_portafoglio = variable_x.sum()                        # v0
mean_portfolio_return = variable_x.dot(mu_j)                # v
portfolio_return_t = np.dot(df.transpose(), variable_x)     # y

#             Vincoli standard
m.addConstr(variable_d.dot(p_t) == 1, name="c18")
m.addConstrs((variable_d[k] >= ((Benchmark[k] + alfa) * somma_portafoglio - portfolio_return_t[k])
              for k in range(T_scenarios)), name="c19")

#             Nuovi vincoli
m.addConstrs((variable_d[g] <= mu_alfa * somma_portafoglio - portfolio_return_t[g] + M * variable_utilde[g]
              for g in range(T_scenarios)), name="c37")
m.addConstrs((variable_d[f] <= M*somma_portafoglio - M*variable_utilde[f] for f in range(T_scenarios)), name="c38")
m.addConstrs((variable_utilde[r] >= 0 for r in range(T_scenarios)), name="c39a")
m.addConstrs((variable_utilde[a] <= M*variable_u[a] for a in range(T_scenarios)), name="c39b")
m.addConstrs((variable_utilde[d] <= somma_portafoglio for d in range(T_scenarios)), name="c40")
m.addConstrs((somma_portafoglio - variable_utilde[t] + M*variable_u[t] <= M for t in range(T_scenarios)), name="c41")

#             Vincoli cardinalità
m.addConstrs((variable_x[z] <= variable_z[z]*M for z in range(N_assets)), name="c27")
m.addConstr(variable_z.sum() <= K, name="c28")

#             Vincoli real_features
m.addConstrs((variable_x[i] <= deltavector[i]*somma_portafoglio for i in range(N_assets)), name="c30")
m.addConstrs((variable_z[z] == 1) >> (variable_x[z] >= epsilon*somma_portafoglio) for z in range(N_assets))

#              Final
obj = mean_portfolio_return - mu_alfa*somma_portafoglio
m.setObjective(obj, GRB.MAXIMIZE)
m.optimize()

#                       Obtaining final values
# for v in m.getVars():
#    print('\t%s\t %g' % (v.VarName, v.X))

x0 = np.zeros(N_assets)
var = m.getVars()
for i in range(N_assets):
    x0[i] = float(var[i].X)
print(x0)
print("il valore somma è: %g" % x0.sum())
print("I valori percentuali sono:\n")
print(x0/x0.sum())

y = []
mylabels = []
for k in range(N_assets):
    if x0[k] > 0.0001:
        y.append(x0[k]/x0.sum())
        mylabels.append(["Asset", k + 1])

print(mylabels)
print(y)
