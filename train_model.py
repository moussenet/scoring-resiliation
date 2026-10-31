import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# 1. Chargement : recopie ici la ligne de ton étape 1
df = pd.read_csv('../data/dataset_assurance_ML.csv')

# 2. Variables
TARGET = 'Résiliation'
num_cols = ['Âge', 'Salaire Annuel (€)', 'Prime Annuelle (€)', 'Ancienneté (mois)',
            'Coeff. Bonus-Malus', 'Nb Sinistres (3 ans)', 'Montant Sinistres (€)',
            'Score Risque (0-100)']
cat_cols = ['Type Contrat', 'Catégorie Prof.', 'Usage Véhicule', 'Dernier Sinistre']
X = df[num_cols + cat_cols]
y = df[TARGET]

# 3. Séparation train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Pipeline Random Forest
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
])
pipeline = Pipeline([
    ('prep', preprocessor),
    ('model', RandomForestClassifier(
        n_estimators=300, max_depth=4, min_samples_leaf=10,
        class_weight='balanced', random_state=42)),
])

# 5. Entraînement et AUC sur le test
pipeline.fit(X_train, y_train)
y_proba = pipeline.predict_proba(X_test)[:, 1]

# 6. Sauvegardes
os.makedirs('models', exist_ok=True)
joblib.dump(pipeline, 'models/pipeline_resiliation.pkl')

meta = {
    'modele': 'Random Forest',
    'auc_test': round(float(roc_auc_score(y_test, y_proba)), 3),
    'num_cols': num_cols,
    'cat_cols': cat_cols,
    'num_ranges': {c: {'min': float(X[c].min()),
                       'max': float(X[c].max()),
                       'median': float(X[c].median())} for c in num_cols},
    'cat_values': {c: sorted(X[c].unique().tolist()) for c in cat_cols},
}
with open('models/metadata.json', 'w', encoding='utf-8') as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print('Modèle et métadonnées sauvegardés. AUC test =', meta['auc_test'])