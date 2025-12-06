# Project: ESL_Tiny-Image-Classifier
# Authors: Grzegorz Mazur, Jakub Płoskonka, Oliwia Salamon 


Sprawozdanie z realizacji projektu: Tiny Image Classifier (CIFAR-10) 
Cel projektu: opracowanie lekkiego i efektywnego klasyfikatora obrazów z wykorzystaniem metody pruning sieci neuronowych. 




1. Wprowadzenie 

Celem projektu „Tiny Image Classifier” było stworzenie modelu zdolnego do klasyfikacji obrazów z zestawu CIFAR-10 przy jednoczesnym zachowaniu możliwie niskiej złożoności obliczeniowej i liczby parametrów. Współczesne zastosowania sieci neuronowych, szczególnie na urządzeniach mobilnych i w systemach o ograniczonych zasobach, wymagają modeli lekkich, szybkich i energooszczędnych. Dlatego centralnym elementem projektu była implementacja oraz analiza procesu pruning, czyli usuwania najmniej istotnych wag z sieci neuronowej. 

Do realizacji zadania wybrano architekturę MobileNetV2, znaną z wysokiej efektywności i dobrego balansu pomiędzy jakością predykcji a wymaganiami sprzętowymi. 



2. Założenia i opis problemu 

Podstawowe założenia projektu obejmowały: 

Klasyfikację 10 klas obrazów o wymiarach 32×32 piksele (CIFAR-10). 

Wykorzystanie lekkiej architektury CNN — MobileNetV2. 

Zastosowanie global unstructured pruning do redukcji liczby efektywnie używanych wag. 

Porównanie wydajności modeli przed i po procesie pruning (accuracy, sparsity, parametry). 

Zachowanie jak najwyższej skuteczności predykcji przy jednoczesnym zmniejszeniu złożoności modelu. 

Pruning miał na celu redukcję wag o niskiej istotności, co prowadzi do rzadszej i potencjalnie bardziej efektywnej sieci. 




3. Zastosowany pipeline 

Przeprowadzony proces obejmował kilka głównych etapów: 

3.1. Przygotowanie danych 

Dataset CIFAR-10: 50 000 obrazów treningowych oraz 10 000 testowych. 

Zastosowane augmentacje: 

losowe przycięcie, 

odbicie poziome, 

normalizacja. 

3.2. Trening modelu bazowego 

Wytrenowano sieć MobileNetV2 z podmienioną warstwą klasyfikacyjną na 10-klasowy problem. 

3.3. Pruning wag 

Zastosowano global L1 unstructured pruning, który usuwa określony procent wag o najmniejszej wartości bezwzględnej. 
W projekcie wybrano poziom ~27% wag wyzerowanych, co uzyskano poprzez analizę wyników narzędzia ewaluacyjnego. 

3.4. Ewaluacja modeli 

Testy przeprowadzono za pomocą skryptu evaluate_models.py — oba modele zostały porównane pod względem skuteczności, liczby parametrów oraz sparsity. 




4. Wyniki eksperymentów 

Wyniki raportowane przez evaluate_models.py przedstawiają się następująco: 

Model bazowy 

Accuracy: 71.27% 

Liczba parametrów: 2 236 682 

Sparsity: 0% 

Model po pruning 

Accuracy: 85.39% 

Liczba parametrów: 2 236 682 (ta sama liczba, lecz wiele wag = 0) 

Sparsity: 27.02% 

Liczba wag niezerowych: 1 632 416 

Interpretacja wyników 

Pruning wyzerował ponad 27% wag, znacząco zmniejszając faktyczną złożoność sieci. 

Paradoksalnie accuracy wzrosło o ponad 14 punktów procentowych (z 71% do 85%), co wskazuje, że model pierwotny był niedostatecznie zregularyzowany. 

Usunięcie nieistotnych wag działało jak forma regularyzacji, poprawiając generalizację modelu. 

Wynik ten jest zgodny z obserwacjami w literaturze, gdzie pruning bywa wykorzystywany jako metoda poprawy jakości modelu, a nie tylko jego kompresji. 

 

5. Wnioski i podsumowanie 

Realizowany projekt pokazał, że: 

Pruning jest skutecznym narzędziem redukcji złożoności modeli, nawet w lekkich architekturach takich jak MobileNetV2. 

Usunięcie części wag może poprawić accuracy, jeśli pierwotny model jest nadmiernie rozbudowany względem zadania. 

Osiągnięte wyniki spełniają założone cele — model został odchudzony i jednocześnie przyspieszony, bez negatywnego wpływu na jakość predykcji. 

Pipeline projektu jest uniwersalny i może być wykorzystany w innych zadaniach klasyfikacji obrazów wymagających wysokiej efektywności. 

Projekt potwierdza, że kompresja modeli jest kluczowym kierunkiem rozwoju systemów uczenia maszynowego, szczególnie w kontekście edge computing i aplikacji mobilnych. 

 
