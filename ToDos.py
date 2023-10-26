
#TODO: should see videos of all movements and afterwards decide which two differ most and then print them and use them to train

# TODO: alternativen für model prediction:
    # empfange emg  daten
    # berechne crosscorrelation zwischen heatmap jetzt und heatmap in unterschiedlichen zeiten vorher
    # zusätzlich mean heatmaps für die unterschiedlichen zeitpunkte der bewegung berechnen (vorher bei channel extraction die mean heatmaps nehmen)
    # und dann die crosscorrelation zwischen dieser heatmap und der den mean heatmaps berechenen.
    # zusätzlich das local model nehmne und auch mit reinrechnen
    #


# TODO: verwende 2d crosscorr um ein gelernten aktivierungsbereich mit heatmpa zu vergleichen

# TODO beim erstellen der trainingsdata: muss ich die anfänge skippen für time data weil sonst heatmap über zu wenige samples geht und das outlier sein könnten
# einfach abfragen ob samples mehr als time:window_samples zur verfügung steht

# TODO remove offset von ref data


# TODO  trainings data abspeichern in realtime anwendung und dann laden
# TODO  trainierte bäume abspeichern und laden und plotten am ende

# TODO in filter.py : gewichte die punkte in der vergangenheit unterschiedliche -> letzte smoothed prediction trägt mehr zu vorhersage bei als die vor 10 mal