# Estimación de potencia en sistemas ESP

Proyecto de machine learning para estimar el consumo promedio de potencia de sistemas de bombeo electrosumergible (ESP, *Electrical Submersible Pumping*).

El modelo utiliza información operacional y de producción para predecir `Average_Kilowatt`. El flujo carga y limpia los datos, prepara las variables, entrena una regresión lineal, calcula RMSE y R², y registra el experimento en MLflow.

## Estructura

```text
esp-project/
├── data/
│   └── raw/
│       └── ESP_Data_wells_Anonymized_Rev0_Sin_Well_Name.csv
├── src/
│   └── esp_project/
│       ├── __init__.py
│       ├── data_cleaning/
│       ├── features/
│       ├── models/
│       ├── pipelines/
│       └── visualization/
├── .python-version
├── MLproject
├── pyproject.toml
├── requirements.txt
└── uv.lock
```

Responsabilidades principales:

- `data_cleaning`: carga el CSV, valida sus columnas y limpia los datos.
- `features`: crea el preprocesamiento numérico y categórico.
- `models`: construye y evalúa la regresión lineal.
- `visualization`: genera la gráfica de valores reales frente a predicciones.
- `pipelines`: coordina el entrenamiento y el registro en MLflow.

## Requisitos

- [uv](https://docs.astral.sh/uv/).
- El proyecto usa Python 3.10.22, definido en `.python-version`.

No es necesario instalar Python ni crear el entorno virtual manualmente: `uv` puede administrarlos.

## Instalación

Desde la raíz del proyecto:

```powershell
uv sync --locked
```

Este comando crea o actualiza `.venv`, instala el proyecto en modo editable y utiliza exactamente las versiones registradas en `uv.lock`.

Si se modifica `pyproject.toml`, actualizar el lock y sincronizar:

```powershell
uv lock
uv sync
```

`pyproject.toml` es la fuente principal de dependencias. `requirements.txt` se conserva como alternativa para herramientas que todavía esperan ese formato.

## Dataset

El archivo esperado es:

```text
data/raw/ESP_Data_wells_Anonymized_Rev0_Sin_Well_Name.csv
```

El CSV utiliza punto y coma (`;`) como separador. Debe contener, como mínimo:

```text
Area
Field
Bench
Average_Kilowatt
```

`Average_Kilowatt` es la variable objetivo.

## Ejecutar el entrenamiento

Con la configuración predeterminada:

```powershell
uv run python -m esp_project.pipelines.training
```

Cuando se ejecuta directamente, el pipeline utiliza `sqlite:///mlflow.db` como almacenamiento local de tracking.

Con parámetros personalizados:

```powershell
uv run python -m esp_project.pipelines.training `
  --test-size 0.25 `
  --impute-strategy mean `
  --fit-intercept true `
  --random-state 100
```

Parámetros disponibles:

| Parámetro | Descripción | Predeterminado |
| --- | --- | --- |
| `--data-path` | Ruta del dataset | `data/raw/ESP_Data_wells_Anonymized_Rev0_Sin_Well_Name.csv` |
| `--test-size` | Proporción reservada para evaluación | `0.20` |
| `--impute-strategy` | `mean`, `median` o `most_frequent` | `median` |
| `--fit-intercept` | Define si la regresión calcula intercepto | `true` |
| `--random-state` | Semilla de la división de datos | `42` |

## Ejecutar como MLflow Project

`MLproject` proporciona una interfaz reproducible para lanzar el mismo pipeline. Antes de usarlo, definir el tracking URI en la terminal actual para que la corrida externa y el entrenamiento escriban en la misma base:

```powershell
$env:MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
uv run mlflow run . --env-manager local
```

Con parámetros personalizados:

```powershell
$env:MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
uv run mlflow run . --env-manager local `
  -P test_size=0.25 `
  -P impute_strategy=mean `
  -P fit_intercept=false `
  -P random_state=100
```

## Consultar experimentos

El pipeline registra:

- Parámetros de entrenamiento.
- RMSE y R².
- Modelo entrenado.
- Predicciones del conjunto de prueba.
- Gráfica de valores reales frente a predicciones.

Iniciar la interfaz local:

```powershell
uv run mlflow ui --backend-store-uri "sqlite:///mlflow.db"
```

Después abrir `http://127.0.0.1:5000`.

Ejecutar este comando desde la raíz del proyecto. MLflow utiliza `mlflow.db` para los metadatos y `mlruns/` para los artefactos. Ambos resultados, junto con `.venv`, están excluidos del control de versiones.

## Flujo del modelo

1. Lee el CSV con separador `;`.
2. Normaliza `Area`, `Field` y `Bench`.
3. Convierte las columnas restantes a números.
4. Elimina duplicados y registros sin variable objetivo.
5. Divide los datos en entrenamiento y prueba.
6. Imputa y escala las variables numéricas con `MinMaxScaler`.
7. Imputa y codifica las categorías con `OneHotEncoder`.
8. Entrena una regresión lineal.
9. Calcula RMSE y R².
10. Registra métricas, predicciones, gráfica y modelo en MLflow.

Una ejecución de referencia con `test_size=0.20` y `random_state=42` produce aproximadamente:

```text
RMSE: 33.63
R²:   0.62
```

## Problemas comunes

### El entorno está desactualizado

```powershell
uv sync --locked
```

### No se encuentra el paquete `esp_project`

Ejecutar los comandos desde la raíz del proyecto y usar `uv run`:

```powershell
uv run python -m esp_project.pipelines.training
```

### No se encuentra el dataset

Verificar la ruta predeterminada o indicar otra:

```powershell
uv run python -m esp_project.pipelines.training --data-path "ruta/al/dataset.csv"
```
