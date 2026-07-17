
def evaluate_mae(gt_measurements,estim_measurements):
    '''
    Compare two sets of measurements - given as dicts - by finding
    the mean absolute error (MAE) of each measurement.
    :param gt_measurement: dict of {measurement:value} pairs
    :param estim_measurements: dict of {measurement:value} pairs

    Returns
    :param errors: dict of {measurement:value} pairs of measurements
                    that are both in gt_measurement and estim_measurements
                    where value corresponds to the mean absoulte error (MAE)
                    in cm
    '''
    # Questa funzione confronta due dizionari di misure corporee:
    # - gt_measurements: le misure "vere" (ground truth), es. misurate a mano o note
    # - estim_measurements: le misure stimate dal modello (es. dalla mesh 3D)
    # Per ogni misura presente in entrambi i dizionari calcola l'errore assoluto,
    # utile per capire quanto è accurata la stima rispetto al valore reale.

    # dizionario che conterrà, per ogni misura, l'errore assoluto calcolato
    MAE = {}

    # scorriamo tutte le misure "vere" (nome della misura, valore)
    for m_name, m_value in gt_measurements.items():
        # consideriamo la misura solo se esiste anche tra le misure stimate
        if m_name in estim_measurements.keys():
            # calcoliamo l'errore assoluto: |valore reale - valore stimato|
            error = abs(m_value - estim_measurements[m_name])
            # salviamo l'errore nel dizionario dei risultati
            MAE[m_name] = error

    # se non abbiamo trovato nessuna misura in comune tra i due dizionari...
    if MAE == {}:
        # ...avvisiamo l'utente che non c'è nulla da confrontare
        print("Measurement dicts do not have any matching measurements!")
        print("Returning empty dict!")

    # restituiamo il dizionario con gli errori assoluti (può essere vuoto)
    return MAE


# questo blocco di esempio/demo viene eseguito solo se il file è lanciato direttamente
# (non quando viene importato come modulo da un altro script)
if __name__ == "__main__":

    # torch: libreria per il calcolo tensoriale, usata qui per generare parametri casuali
    import torch
    # pandas: libreria per gestire tabelle di dati (DataFrame), usata per stampare i risultati
    import pandas as pd
    # MeasureSMPL: classe che sa calcolare le misure corporee a partire da un modello SMPL
    from measure import MeasureSMPL
    # MeasurementDefinitions: contiene l'elenco delle misure disponibili/calcolabili
    from measurement_definitions import MeasurementDefinitions

    # percorso della cartella con i dati del modello SMPL
    smpl_path = "/SMPL-Anthropometry/data/SMPL"

    # creiamo un primo "misuratore" basato sul modello SMPL
    measurer1 = MeasureSMPL(smpl_path=smpl_path)
    # generiamo 10 parametri di forma (beta) casuali con distribuzione normale
    # (media 0, deviazione standard 1) per creare un corpo casuale di esempio
    betas1 = torch.empty((1,10)).normal_(mean=0,std=1)
    # costruiamo la mesh 3D del corpo a partire dai parametri di forma generati
    measurer1.from_smpl(gender="MALE", shape=betas1)

    # creiamo un secondo "misuratore", indipendente dal primo
    measurer2 = MeasureSMPL(smpl_path=smpl_path)
    # generiamo un altro set casuale di parametri di forma per il secondo corpo
    betas2 = torch.empty((1,10)).normal_(mean=0,std=1)
    # costruiamo la mesh 3D del secondo corpo
    measurer2.from_smpl(gender="MALE", shape=betas2)


    # otteniamo l'elenco di tutte le misure possibili definite dal progetto
    measurement_names = MeasurementDefinitions.possible_measurements
    # calcoliamo effettivamente tutte quelle misure sul primo corpo
    measurer1.measure(measurement_names)
    # e sul secondo corpo
    measurer2.measure(measurement_names)


    # confrontiamo le misure dei due corpi calcolando il MAE tra loro
    # (qui è solo un esempio dimostrativo: normalmente si confronterebbero misure
    # reali/note con misure stimate, non due corpi generati casualmente)
    MAE = evaluate_mae(measurer1.measurements,measurer2.measurements)
    # costruiamo una tabella (DataFrame) con il nome della misura e il relativo errore
    mae_table = pd.DataFrame({"Measurement":MAE.keys(),
                              "MAE(cm)": MAE.values()})
    # stampiamo la tabella a schermo
    print(mae_table)
