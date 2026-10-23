# API Heart MLOps

API REST (FastAPI) que predice el riesgo de enfermedad cardiaca de un paciente. La API no carga el modelo: reenvía los datos al endpoint **`heart-mlops-endpoint`** de Databricks Model Serving, que sirve `workspace.heart_mlops.heart_disease_model@champion` (entrenado en NB02 y promovido en NB03 con el dataset `heart_disease_uci`).

```
Usuario ──► API FastAPI (Azure Container Apps: api-heart-mlops) ──► Databricks Model Serving (heart-mlops-endpoint) ──► modelo @champion
```

| Archivo | Para qué sirve |
|---|---|
| `main.py` | La API: `/health` y `/predecir` (valida las 11 variables del modelo) |
| `requirements.txt` | Librerías de la API (versiones fijas) |
| `requirements-dev.txt` | Librerías de pruebas (pytest, httpx) |
| `Dockerfile` | Receta de la imagen que se ejecuta en Azure |
| `.github/workflows/deploy.yml` | CI/CD: pruebas, imagen al ACR y despliegue en Container Apps |
| `tests/test_api.py` | Pruebas automáticas (no llaman a Databricks) |
| `paciente.json` | Cuerpo de prueba para el **endpoint de Databricks** (Fase 1) |
| `paciente_api.json` | Cuerpo de prueba para la **API** (`/predecir`) |

Seguridad: la URL y el token de Databricks y las credenciales de Azure solo viven en **GitHub Secrets** y en los secretos de la Container App. Nunca se escriben en el código. El CSV del dataset no se sube al repositorio.

## Contrato de entrada de `/predecir`

Exactamente estas 11 claves, ni más ni menos (una clave extra o faltante devuelve **422**):

| Variable | Tipo | Valores permitidos | ¿Acepta `null`? |
|---|---|---|---|
| `age` | número | 18 a 100 | No |
| `trestbps` | número | 80 a 220 | Sí |
| `chol` | número | 50 a 700 | Sí |
| `thalch` | número | 60 a 220 | Sí |
| `oldpeak` | número | −3 a 7 | Sí |
| `fbs` | entero | 0 o 1 | Sí |
| `exang` | entero | 0 o 1 | Sí |
| `sex` | texto | `Male`, `Female` | No |
| `cp` | texto | `typical angina`, `atypical angina`, `non-anginal`, `asymptomatic` | No |
| `restecg` | texto | `normal`, `st-t abnormality`, `lv hypertrophy` | Sí |
| `slope` | texto | `upsloping`, `flat`, `downsloping` | Sí |

`null` se permite solo donde el dataset original tenía faltantes: el Pipeline del modelo los imputa igual que en el entrenamiento. Respuesta:

```json
{"predictions": [1], "resultado": "Riesgo alto de enfermedad cardiaca",
 "aviso": "Resultado de un modelo académico; no reemplaza un diagnóstico médico."}
```

## Ejecutar en local (Windows, PowerShell)

```powershell
cd C:\heart-mlops-api
python -m pip install -r requirements-dev.txt
python -m pytest -q
$env:DATABRICKS_URL = "https://TU_WORKSPACE.cloud.databricks.com/serving-endpoints/heart-mlops-endpoint/invocations"
$env:DATABRICKS_TOKEN = "dapi..."
python -m uvicorn main:app --reload
```

Abre http://127.0.0.1:8000/docs para probar `/predecir` desde el navegador.

## Despliegue

El workflow `deploy.yml` se ejecuta en cada push a `main`. Reutiliza `rg-mlops-ean`, `env-mi-proyecto` y `acrmiprojectolaura26` del Módulo 3 y crea (o actualiza) la Container App **`api-heart-mlops`**. Necesita 5 secretos en *Settings → Secrets and variables → Actions*:

| Secreto | Valor |
|---|---|
| `AZURE_CREDENTIALS` | JSON de 4 campos de la cuenta de servicio (clientId, clientSecret, subscriptionId, tenantId) |
| `ACR_USERNAME` | Usuario admin del ACR (`acrmiprojectolaura26`) |
| `ACR_PASSWORD` | Contraseña admin del ACR |
| `DATABRICKS_URL` | `https://TU_WORKSPACE.cloud.databricks.com/serving-endpoints/heart-mlops-endpoint/invocations` |
| `DATABRICKS_TOKEN` | Token personal de Databricks (`dapi...`) |

Al terminar, el paso *URL de la API y prueba de humo* imprime la URL pública y comprueba `/health`.
