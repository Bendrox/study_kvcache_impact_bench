# Étude du KV Caching : Accélération de la Génération Autorégressive

Ce projet aborde sans prétention le sujet du **KV Caching** dans les modèles de langage (LLM) basés sur une architecture Transformer (décodeur). 
L'ojectif est de comprendre l'interet de ce mécanisme et de mesurer son impact a travers quelques benchmarks sur GPT-2 small (124M paramètres) à la fois sur le temps de génération (des tokens) et la consommation de la mémoire.

## 📌 Objectifs de l'étude
* Comprendre le fonctionnement du KV cache lors de la génération de tokens.
* Comparer une approche naïve (recalcul total à chaque étape) avec une approche optimisée par cache.
* Mesurer les gains de performance (vitesse et accélération) en fonction de la longueur de la séquence générée.

## 🚀 État actuel du projet (`study.ipynb`)
Le notebook principal couvre  :
* **La théorie :** Explication du coût quadratique de la génération standard ($O(T^{3})$ pour la séquence totale) et de la réduction mathématique apportée par le cache ($O(T^{2})$).
* **L'implémentation Naïve :** Génération autorégressive sans cache (`use_cache=False`), forçant le modèle à recalculer l'attention sur tout le contexte à chaque étape.
* **L'implémentation avec Cache :** Utilisation de `past_key_values` de Hugging Face pour stocker les clés et les valeurs des tokens précédents.
* **Benchmarks CPU :** Mesures chronométrées montrant une accélération claire et proportionnelle à la longueur de la séquence générée (jusqu'à x8.1 pour 1000 tokens).
* **Benchmarks GPU (Limites observées) :** Mise en évidence d'une courbe différente sur GPU, suggérant que la parallélisation matérielle masque en partie l'efficacité du KV Cache sur des petites tailles de batch/séquences.

## 🛠️ Reste à faire :

- [ ] Rajouter le monitoring de la VRAM / mémoire utilisée lors des générations.
- [ ] Rajouter des explications et commentaires sur les goulots d'étranglement : *Memory Bound* vs *Compute/Latency Bound*.
- [ ] Approfondir le benchmark GPU avec des limites plus poussées (gros batch size, très longues séquences) pour rendre le gain du cache visible.
- [ ] Séparer explicitement les phases de **Prefill** (traitement du prompt) et de **Decode** (génération token par token) dans le code et les mesures.
- [ ] Implémenter un KV Cache manuellement (sans utiliser `past_key_values` natif de HF) pour bien en comprendre la structure interne.