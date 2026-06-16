"""Chargement et partitionnement des donnees."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from mlproject.config import DATA_PATH, RANDOM_STATE, TARGET

# Mapping des valeurs textuelles de la colonne cible vers 0/1
_TARGET_MAP = {"Approved": 1, "Rejected": 0}


def load_data(path=None) -> pd.DataFrame:
    """Charger le dataset CSV et normaliser la colonne cible.

    La colonne ``loan_status`` du dataset Kaggle contient des valeurs
    avec espaces ("  Approved", " Rejected") et doit etre convertie en 0/1.
    La colonne ``loan_id`` est un identifiant sans valeur predictive et
    est supprimee ici.

    Parameters
    ----------
    path : path-like, optional
        Chemin vers le fichier CSV. Par defaut, ``DATA_PATH`` de la config.

    Returns
    -------
    pd.DataFrame
        DataFrame pret a l'emploi avec la cible encodee en entier (0/1).
    """
    df = pd.read_csv(path or DATA_PATH)

    # Supprimer l'identifiant non predictif
    df = df.drop(columns=["loan_id"], errors="ignore")

    # Normaliser les noms de colonnes (espaces eventuels)
    df.columns = df.columns.str.strip()

    # Encoder la cible texte -> 0/1
    df[TARGET] = df[TARGET].str.strip().map(_TARGET_MAP)

    return df


def split(
    df: pd.DataFrame,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separer features / cible puis creer les jeux d'entrainement et de test.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame complet (features + cible).
    test_size : float, optional
        Proportion du jeu de test, par defaut 0.2.

    Returns
    -------
    tuple
        x_train, x_test, y_train, y_test
    """
    x = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(
        x, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )
