# Pruebas de la API sin llamar a Databricks: se simula (mock) la respuesta del endpoint.
# Ejecutar con:  pip install -r requirements-dev.txt  y luego  pytest -q
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DATABRICKS_URL", "https://ejemplo/serving-endpoints/heart-mlops-endpoint/invocations")
os.environ.setdefault("DATABRICKS_TOKEN", "token-de-prueba")

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402

client = TestClient(main.app)

PACIENTE = {
    "age": 63, "trestbps": 145, "chol": 233, "thalch": 150, "oldpeak": 2.3,
    "fbs": 1, "exang": 0, "sex": "Male", "cp": "typical angina",
    "restecg": "lv hypertrophy", "slope": "downsloping",
}


class RespuestaFalsa:
    status_code = 200
    text = '{"predictions": [1]}'

    def json(self):
        return {"predictions": [1]}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predecir_envia_las_11_variables(monkeypatch):
    enviado = {}

    def post_falso(url, headers, json, timeout):
        enviado.update(json)
        return RespuestaFalsa()

    monkeypatch.setattr(main.requests, "post", post_falso)
    r = client.post("/predecir", json=PACIENTE)
    assert r.status_code == 200
    assert r.json()["predictions"] == [1]
    registro = enviado["dataframe_records"][0]
    assert sorted(registro) == sorted(PACIENTE)          # exactamente las 11 variables
    assert isinstance(registro["age"], float)            # la firma pide double
    assert isinstance(registro["fbs"], float)


def test_campo_extra_es_rechazado():
    r = client.post("/predecir", json={**PACIENTE, "dataset": "Cleveland"})
    assert r.status_code == 422


def test_campo_faltante_es_rechazado():
    sin_chol = {k: v for k, v in PACIENTE.items() if k != "chol"}
    assert client.post("/predecir", json=sin_chol).status_code == 422


def test_valores_fuera_de_rango_o_categoria_invalida():
    assert client.post("/predecir", json={**PACIENTE, "age": 150}).status_code == 422
    assert client.post("/predecir", json={**PACIENTE, "cp": "dolor fuerte"}).status_code == 422
    assert client.post("/predecir", json={**PACIENTE, "fbs": 2}).status_code == 422


def test_nulos_permitidos_en_variables_con_faltantes(monkeypatch):
    monkeypatch.setattr(main.requests, "post", lambda *a, **k: RespuestaFalsa())
    con_nulos = {**PACIENTE, "chol": None, "slope": None, "fbs": None}
    assert client.post("/predecir", json=con_nulos).status_code == 200
