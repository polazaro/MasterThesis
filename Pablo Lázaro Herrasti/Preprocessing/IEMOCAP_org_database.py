import shutil, os


for i in range(5):
    
    print(i+1)
    
    ruta = "C:/Users/ruben/Documents/Máster Data Science/2º Cuatrimestre/Master Thesis/Database/IEMOCAP_full_release/Session"+str(i+1)+"/dialog/"
    origen_video = ruta + 'avi/DivX/'
    destino_video = "C:/Users/ruben/Documents/Máster Data Science/2º Cuatrimestre/Master Thesis/Database/IEMOCAP_windows/Video/Session"+str(i+1)
    origen_audio = ruta + 'wav/'
    destino_audio = "C:/Users/ruben/Documents/Máster Data Science/2º Cuatrimestre/Master Thesis/Database/IEMOCAP_windows/Audio/Session"+str(i+1)
    origen_labels = ruta + 'EmoEvaluation/'
    destino_labels = "C:/Users/ruben/Documents/Máster Data Science/2º Cuatrimestre/Master Thesis/Database/IEMOCAP_windows/Labels/Session"+str(i+1)
    files = os.listdir(origen_video)
    
    for j in range(len(files)):
        file = origen_video+"/"+files[j]
        shutil.copy(file, destino_video)
    files = os.listdir(origen_audio)
    for j in range(len(files)):
        file = origen_audio+"/"+files[j]
        shutil.copy(file, destino_audio)
    files = os.listdir(origen_labels)
    for j in range(8,len(files)):
        file = origen_labels+"/"+files[j]
        shutil.copy(file, destino_labels)