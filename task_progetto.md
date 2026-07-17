# PROGETTO 4 – Simulatore di avatar umani realistici da modelli SMPL

# per virtual try-on

**Background**
I modelli parametrici del corpo umano, come SMPL e le sue estensioni, rappresentano una
base molto solida per la modellazione digitale del corpo, poiché consentono di descrivere
forma, posa e proporzioni corporee in modo controllabile e coerente. Tuttavia, nelle loro
forme standard, questi modelli risultano spesso troppo astratti o poco leggibili dal punto di
vista visivo per applicazioni che richiedono una rappresentazione più credibile della
persona, come simulazione corporea, visualizzazione personalizzata, human-centered XR e
virtual try-on. In molti scenari applicativi, infatti, non è sufficiente disporre di un body model
parametrico: è necessario costruire una pipeline capace di generare avatar visualizzabili,
consistenti con le misure e con le caratteristiche antropometriche dell’utente reale,
mantenendo al tempo stesso la compatibilità con pose, animazioni e successive
elaborazioni. Questo aspetto è particolarmente rilevante quando l’avatar deve diventare la
base per applicazioni di prova virtuale di capi o accessori, in cui coerenza morfologica,
proporzioni e controllabilità del modello sono essenziali.

Il progetto si colloca quindi nell’ambito di human modeling, computer graphics, avatar
generation e digital human simulation, con l’obiettivo di costruire un sistema per la
generazione di avatar umani parametrizzati e riutilizzabili in futuri scenari di virtual try-on,
visualizzabili e coerenti con i parametri corporei e con le misure della persona reale.

**Task**
● Interfaccia di definizione e gestione dei parametri corporei dell’utente, con
particolare attenzione a misure, proporzioni e caratteristiche
antropometriche
● Generazione di un avatar umano visualizzabile, leggibile e coerente con i dati
della persona reale
● Possibilità di controllare forma del corpo, posa e configurazione del modello
con renderizzazioni 2D

**Criterio di valutazione**

```
− Coerenza tra parametri corporei in input e avatar generato
− Accuratezza della rappresentazione delle proporzioni e delle misure della persona
reale
− Correttezza nella gestione dei parametri, della posa e dell’eventuale esportazione
del modello
− Qualità della presentazione del lavoro svolto e dei risultati ottenuti
```
**Consegna**


- PowerPoint per la presentazione del lavoro svolto
- Video dimostrativo
- Codice sorgente del sistema
- Documentazione su sviluppo e utilizzo del codice
**Piattaforma**
- Unity
- Python

**Materiale a disposizione:**

- https://github.com/DavidBoja/SMPL-Anthropometry
- https://sdh.global/projects/virtual-try-on-room-with-smpl-anthropometry/
- https://github.com/mkocabas/body-model-visualizer