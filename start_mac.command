#!/bin/bash
cd "$(dirname "$0")"
if [ ! -f .venv/installed.ok ]; then
  echo "Πρώτη εκκίνηση: εγκατάσταση (1-3 λεπτά)..."
  python3 -m venv .venv || { echo "Δεν βρέθηκε Python 3. Κατέβασέ το από https://www.python.org/downloads/"; read -r; exit 1; }
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt || { echo "Η εγκατάσταση απέτυχε. Έλεγξε τη σύνδεση και ξαναδοκίμασε."; read -r; exit 1; }
  touch .venv/installed.ok
fi
echo "Η εφαρμογή ανοίγει στον browser. Κλείσε αυτό το παράθυρο για να τη σταματήσεις."
.venv/bin/python -m streamlit run app.py
