from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix

from practica2_aplic_algoritmos_ml.fase2_preparacion_de_datos import X_train_scaled, y_train, X_test_scaled, y_test

modelo = LogisticRegression(max_iter=1000,
multi_class='multinomial',

solver='lbfgs', random_state=42)

modelo.fit(X_train_scaled, y_train)
y_pred = modelo.predict(X_test_scaled)
y_proba = modelo.predict_proba(X_test_scaled)
print("=== EXACTITUD ===")
print(f"Accuracy Train: {modelo.score(X_train_scaled,
y_train):.3f}")
print(f"Accuracy Test: {modelo.score(X_test_scaled, y_test):.3f}")
print("\n=== REPORTE DE CLASIFICACIÓN ===")
print(classification_report(y_test, y_pred))
print("\n=== MATRIZ DE CONFUSIÓN ===")
cm = confusion_matrix(y_test, y_pred)
print(cm)