import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms
from PIL import Image
import time
import warnings
import traceback

# Ignorar warnings específicos
warnings.filterwarnings("ignore", category=UserWarning)

# Para reproducibilidad
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ====== CONFIGURACIÓN ======
BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 0.001
IMAGE_SIZE = 224
TRAIN_CSV = "data/train.csv"
TRAIN_DIR = "data/train/train"
TEST_CSV = 'data/sample_submission.csv'
TEST_DIR = 'data/test/test'
SAVE_DIR = "resultados"
MODEL_NAME = "mejor_modelo.pth"

# ====== CLASES DATASET ======
class FrutasDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.datos = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.clases = sorted(self.datos['Label'].unique())
        self.clase_a_idx = {cls: i for i, cls in enumerate(self.clases)}
        
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
            
        etiqueta = self.datos.iloc[idx, 1]
        etiqueta_idx = self.clase_a_idx[etiqueta]
        
        if self.transform:
            imagen = self.transform(imagen)
            
        return imagen, etiqueta_idx

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

# ====== FUNCIONES DE UTILIDAD ======
def crear_modelo(num_clases):
    """Crea un modelo ResNet34 con la capa final ajustada"""
    modelo = models.resnet34(weights='DEFAULT')
    # Congelar todas las capas
    for param in modelo.parameters():
        param.requires_grad = False
    # Reemplazar la capa final
    num_features = modelo.fc.in_features
    modelo.fc = nn.Linear(num_features, num_clases)
    return modelo

def visualizar_muestras(dataset, clases, num_imagenes=6):
    """Visualiza algunas imágenes de muestra del dataset con sus etiquetas"""
    # Crear figura
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    # Seleccionar índices aleatorios
    indices = np.random.choice(len(dataset), num_imagenes, replace=False)
    
    for i, idx in enumerate(indices):
        img, label_idx = dataset[idx]
        # Convertir tensor a imagen
        img = img.permute(1, 2, 0).numpy()
        # Desnormalizar
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        # Mostrar imagen
        axes[i].imshow(img)
        axes[i].set_title(f'Clase: {clases[label_idx]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'muestras_dataset.png'))
    plt.close()

def graficar_metricas(train_loss, val_loss, train_acc, val_acc):
    """Genera gráficos de pérdida y precisión del entrenamiento"""
    epochs = range(1, len(train_loss) + 1)
    
    # Gráfico de pérdida
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title('Loss durante el entrenamiento')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Gráfico de precisión
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    plt.title('Precisión durante el entrenamiento')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'metricas_entrenamiento.png'))
    plt.close()

# ====== FUNCIÓN DE ENTRENAMIENTO ======
def entrenar():
    print("Iniciando entrenamiento de clasificación de frutas...")
    
    # Crear directorio para resultados
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Configurar dispositivo (CPU o GPU si está disponible)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # Transformaciones para imágenes
    transformacion = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Cargar datos
    print("Cargando dataset...")
    try:
        dataset = FrutasDataset(
            csv_file=TRAIN_CSV,
            img_dir=TRAIN_DIR,
            transform=transformacion
        )
        print(f"Clases encontradas: {dataset.clases}")
        
        # Visualizar algunas muestras del dataset
        try:
            visualizar_muestras(dataset, dataset.clases)
            print("Muestras del dataset guardadas en", os.path.join(SAVE_DIR, 'muestras_dataset.png'))
        except Exception as e:
            print(f"No se pudieron visualizar muestras: {e}")
        
        # Dividir en entrenamiento y validación
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        indices = list(range(len(dataset)))
        np.random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Crear subconjuntos
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        
        print(f"Total de imágenes: {len(dataset)}")
        print(f"Imágenes para entrenamiento: {len(train_dataset)}")
        print(f"Imágenes para validación: {len(val_dataset)}")
        
        # Crear dataloaders SIN MULTIPROCESAMIENTO
        train_loader = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )
        
        # Crear modelo
        print("Creando modelo ResNet34...")
        model = crear_modelo(len(dataset.clases))
        
        # Mover a dispositivo y configurar entrenamiento
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
        
        # Entrenamiento
        print("Comenzando entrenamiento...")
        start_time = time.time()
        mejor_acc = 0.0
        
        # Para graficar métricas
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        for epoch in range(NUM_EPOCHS):
            # Modo entrenamiento
            model.train()
            running_loss = 0.0
            correctos = 0
            total = 0
            
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Zero gradientes
                optimizer.zero_grad()
                
                # Forward + backward
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                # Estadísticas
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correctos += (predicted == labels).sum().item()
                
                # Mostrar progreso
                if (i+1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Batch {i+1}/{len(train_loader)}, "
                          f"Loss: {loss.item():.4f}, "
                          f"Acc: {100 * correctos / total:.2f}%")
            
            # Calcular pérdida y precisión de entrenamiento
            epoch_loss = running_loss / len(train_dataset)
            epoch_acc = 100 * correctos / total
            
            # Validación
            model.eval()
            val_loss = 0.0
            val_correctos = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correctos += (predicted == labels).sum().item()
            
            # Calcular pérdida y precisión de validación
            val_loss = val_loss / len(val_dataset)
            val_acc = 100 * val_correctos / val_total
            
            # Guardar métricas
            history['train_loss'].append(epoch_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(epoch_acc)
            history['val_acc'].append(val_acc)
            
            # Imprimir resultados
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} - "
                  f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}% - "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Guardar mejor modelo
            if val_acc > mejor_acc:
                mejor_acc = val_acc
                
                # Guardar el modelo
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'clases': dataset.clases,
                    'clase_a_idx': dataset.clase_a_idx,
                    'val_acc': val_acc
                }
                
                torch.save(checkpoint, os.path.join(SAVE_DIR, MODEL_NAME))
                print(f"Modelo guardado con precisión: {val_acc:.2f}%")
        
        # Tiempo total
        tiempo_total = time.time() - start_time
        print(f"Entrenamiento completado en {tiempo_total/60:.2f} minutos")
        print(f"Mejor precisión de validación: {mejor_acc:.2f}%")
        
        # Graficar métricas del entrenamiento
        graficar_metricas(
            history['train_loss'], 
            history['val_loss'], 
            history['train_acc'], 
            history['val_acc']
        )
        print(f"Gráficos de entrenamiento guardados en {SAVE_DIR}")
        
        return True
        
    except Exception as e:
        print(f"Error en entrenamiento: {e}")
        traceback.print_exc()
        return False

# ====== FUNCIÓN DE PREDICCIÓN ======
def predecir():
    print("Iniciando predicción de clasificación de frutas...")
    
    # Crear directorio para resultados
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Configurar dispositivo (CPU o GPU si está disponible)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # Verificar archivos
    modelo_path = os.path.join(SAVE_DIR, MODEL_NAME)
    if not os.path.exists(modelo_path):
        print(f"Error: No se encontró el modelo en {modelo_path}")
        return False
    
    if not os.path.exists(TEST_CSV):
        print(f"Error: No se encontró el archivo CSV de test en {TEST_CSV}")
        return False
    
    # Cargar modelo
    print(f"Cargando modelo desde {modelo_path}...")
    try:
        checkpoint = torch.load(modelo_path, map_location=device)
        
        # Obtener clases
        if 'clases' in checkpoint:
            clases = checkpoint['clases']
        elif 'clase_a_idx' in checkpoint:
            clase_a_idx = checkpoint['clase_a_idx']
            clases = sorted(clase_a_idx, key=clase_a_idx.get)
        else:
            print("Error: No se encontraron clases en el checkpoint")
            return False
        
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
        
        return True
        
    except Exception as e:
        print(f"Error en predicción: {e}")
        traceback.print_exc()
        return False

# ====== FUNCIÓN PRINCIPAL ======
def main():
    parser = argparse.ArgumentParser(description='Clasificación de Frutas con PyTorch')
    parser.add_argument('accion', type=str, choices=['train', 'predict', 'all'], 
                        help='Acción a realizar: train (entrenamiento), predict (predicción) o all (ambos)')
    
    args = parser.parse_args()
    
    if args.accion == 'train':
        entrenar()
    elif args.accion == 'predict':
        predecir()
    elif args.accion == 'all':
        if entrenar():
            print("\n" + "="*50 + "\n")
            predecir()
        else:
            print("No se pudo continuar con la predicción debido a errores en el entrenamiento.")

if __name__ == "__main__":
    main() 