import re
from collections import defaultdict


def index_molecular_formula(mf):
    # Utilisation d'une expression régulière pour trouver les éléments et leurs quantités
    elements = re.findall(r'([A-Z][a-z]?)(\d*)', mf)

    # Dictionnaire pour stocker le nombre d'atomes par élément
    element_counts = defaultdict(int)
    for element, count in elements:
        if count == '':
            count = 1
        else:
            count = int(count)
        element_counts[element] += count

    # Liste pour stocker les résultats
    result = []
    index = 1
    carbon_indices = []  # Liste pour stocker les indices des carbones

    # Indexer les carbones en premier
    if 'C' in element_counts:
        for i in range(element_counts['C']):
            result.append(f"{index} C")
            carbon_indices.append(index)
            index += 1
        del element_counts['C']  # Supprimer après traitement

    # Ignorer les hydrogènes
    if 'H' in element_counts:
        del element_counts['H']

    # Indexer les autres éléments
    for element in sorted(element_counts):
        for i in range(element_counts[element]):
            result.append(f"{index} {element}")
            index += 1

    return result, len(carbon_indices)