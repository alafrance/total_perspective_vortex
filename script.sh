# Ajouter toutes les dépendances
cat requirements.txt | cut -d'=' -f1 | xargs poetry add

# OU une par une (plus sûr)
while read requirement; do
    poetry add "${requirement%==*}" || echo "Failed to add $requirement"
done < requirements.txt