# Scoring de résiliation — Assurance Auto

Application web qui estime la probabilité qu'un client d'assurance auto résilie
son contrat, à partir d'un modèle Random Forest (scikit-learn) et d'une
interface Streamlit.

## Structure du projet

```
app.py                  Interface Streamlit
train_model.py          Entraînement et sauvegarde du modèle
models/                 Pipeline (.pkl) et métadonnées (.json)
data/                   Jeu de données
.streamlit/config.toml  Thème de l'application
requirements.txt        Dépendances figées
```

## Lancer l'application

```
pip install -r requirements.txt
streamlit run app.py
```