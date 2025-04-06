import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import time
import warnings
import traceback

# Ignorar warnings específicos
warnings.filterwarnings("ignore", category=UserWarning)

# ====== CONFIGURACIÓN ======
BATCH_SIZE = 16
IMAGE_SIZE = 224
TEST_CSV = 'data/sample_submission.csv'
TEST_DIR = 'data/test/test'
MODELO_PATH = 'resultados/mejor_modelo.pth'
SAVE_DIR = 'resultados'

# ====== CLASE DATASET ======
class TestDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.datos = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.datos)
    
    def __getitem__(self, idx):
        img_name = self.datos.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Cargar imagen con manejo de errores
        try:
            imagen = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error cargando {img_path}: {e}")
            imagen = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
            
        if self.transform:
            imagen = self.transform(imagen)
            
        return imagen, img_name

def crear_modelo(num_clases):
    """Crea un modelo ResNet34 con la capa final ajustada"""
    modelo = models.resnet34(weights=None)
    num_features = modelo.fc.in_features
    modelo.fc = nn.Linear(num_features, num_clases)
    return modelo

def main():
    print("Iniciando predicción de clasificación de frutas...")
    
    # Crear directorio para resultados
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Configurar dispositivo (CPU o GPU si está disponible)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # Verificar archivos
    if not os.path.exists(MODELO_PATH):
        print(f"Error: No se encontró el modelo en {MODELO_PATH}")
        return
    
    if not os.path.exists(TEST_CSV):
        print(f"Error: No se encontró el archivo CSV de test en {TEST_CSV}")
        return
    
    # Cargar modelo
    print(f"Cargando modelo desde {MODELO_PATH}...")
    try:
        checkpoint = torch.load(MODELO_PATH, map_location=device)
        
        # Obtener clases
        if 'clases' in checkpoint:
            clases = checkpoint['clases']
        elif 'clase_a_idx' in checkpoint:
            clase_a_idx = checkpoint['clase_a_idx']
            clases = sorted(clase_a_idx, key=clase_a_idx.get)
        else:
            print("Error: No se encontraron clases en el checkpoint")
            return
        
        print(f"Clases encontradas: {clases}")
        
        # Crear modelo
        modelo = crear_modelo(len(clases))
        modelo.load_state_dict(checkpoint['model_state_dict'])
        modelo.to(device)
        modelo.eval()
        
        # Transformaciones para imágenes de test
        transformacion = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Cargar dataset de test
        print("Cargando imágenes de test...")
        dataset = TestDataset(
            csv_file=TEST_CSV,
            img_dir=TEST_DIR,
            transform=transformacion
        )
        
        # Crear dataloader SIN MULTIPROCESAMIENTO
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )
        
        print(f"Total de imágenes de test: {len(dataset)}")
        
        # Realizar predicciones
        print("Realizando predicciones...")
        start_time = time.time()
        
        todas_predicciones = []
        todos_archivos = []
        
        with torch.no_grad():
            for i, (imagenes, nombres) in enumerate(dataloader):
                imagenes = imagenes.to(device)
                
                # Forward pass
                salidas = modelo(imagenes)
                _, predicciones = torch.max(salidas, 1)
                
                # Guardar resultados
                todas_predicciones.extend(predicciones.cpu().numpy())
                todos_archivos.extend(nombres)
                
                # Mostrar progreso
                if (i+1) % 10 == 0:
                    print(f"Procesados {i+1}/{len(dataloader)} lotes")
        
        # Convertir índices a nombres de clases
        etiquetas_predichas = [clases[idx] for idx in todas_predicciones]
        
        # Crear DataFrame de resultados
        resultados = pd.DataFrame({
            'Id': todos_archivos,
            'Label': etiquetas_predichas
        })
        
        # Guardar resultados
        output_path = os.path.join(SAVE_DIR, 'submission.csv')
        resultados.to_csv(output_path, index=False)
        
        # Tiempo total
        tiempo_total = time.time() - start_time
        print(f"Predicciones completadas en {tiempo_total:.2f} segundos")
        print(f"Resultados guardados en {output_path}")
        
        # Mostrar estadísticas
        conteo_clases = resultados['Label'].value_counts()
        print("\nDistribución de predicciones:")
        for clase, conteo in conteo_clases.items():
            porcentaje = 100 * conteo / len(resultados)
            print(f"{clase}: {conteo} ({porcentaje:.1f}%)")
        
        # Visualizar distribución
        plt.figure(figsize=(10, 6))
        sns.barplot(x=conteo_clases.index, y=conteo_clases.values)
        plt.title('Distribución de predicciones por clase')
        plt.xlabel('Clase')
        plt.ylabel('Número de predicciones')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, 'distribucion_predicciones.png'))
        plt.close()
        print(f"Gráfico de distribución guardado en {SAVE_DIR}")
        
    except Exception as e:
        print(f"Error en predicción: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 