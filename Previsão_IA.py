import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df = pd.read_csv("advertising.csv")

ia1 = RandomForestRegressor()
ia2 = LinearRegression()

x = df.drop("Vendas", axis=1)
y = df["Vendas"]

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state=1)
ia1.fit(x_treino,y_treino)
ia2.fit(x_treino,y_treino)

result_ia1 = ia1.predict(x_teste)
result_ia2 = ia2.predict(x_teste)

porcentagem_ia1 = metrics.r2_score(y_teste, result_ia1)
porcentagem_ia2 = metrics.r2_score(y_teste, result_ia2)

print(f"Certeza da IA1: {porcentagem_ia1:.2%}")
print(f"Certeza da IA2: {porcentagem_ia2:.2%}")

val_prox_mes = pd.read_csv("novos.csv")
previsao = ia1.predict(val_prox_mes)
val_prox_mes["Vendas"] = previsao
val_prox_mes.to_excel("vendasnovas.xlsx", index=False)
print(val_prox_mes)
