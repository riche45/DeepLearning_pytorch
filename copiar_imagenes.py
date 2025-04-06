import os
import shutil
import pandas as pd

# Directorios
TRAIN_CSV = 'data/train.csv'
TRAIN_DIR = 'data/train/train'
OUTPUT_DIR = 'imagenes_muestra'

# Crear el directorio principal
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Leer el CSV
df = pd.read_csv(TRAIN_CSV)
print(f"Clases: {df['Label'].unique()}")

# Crear directorios para cada clase
for label in df['Label'].unique():
    os.makedirs(os.path.join(OUTPUT_DIR, label), exist_ok=True)
    
    # Obtener 2 imágenes de cada clase
    ejemplos = df[df['Label'] == label].iloc[:2]
    
    for _, row in ejemplos.iterrows():
        src = os.path.join(TRAIN_DIR, row['Id'])
        dst = os.path.join(OUTPUT_DIR, label, row['Id'])
        
        if os.path.exists(src):
            print(f"Copiando {src} a {dst}")
            shutil.copy(src, dst)
        else:
            print(f"Error: No se encontró {src}")
            
print("Proceso completado.") 