import csv

from fastapi.testclient import TestClient

from src.api import app
from src.clean import spanish_keys, spanish_senses
from src.evaluate import character_score, outcome
from src.paths import PAIRS

client = TestClient(app)


def test_la_glosa_descarta_etiquetas_y_notas_morfologicas():
    assert spanish_senses("— v_t") == []
    assert spanish_senses("*v_i") == []
    assert spanish_senses("-in") == []
    assert spanish_senses("n_pos") == []
    assert spanish_senses("estar enfermo -ma") == ["estar enfermo"]
    assert spanish_senses("hacer") == ["hacer"]
    assert spanish_senses("aa'u") == []
    assert spanish_senses("estar alegre; estar contento -ta; estar feliz") == [
        "estar alegre",
        "estar contento",
        "estar feliz",
    ]


def test_una_glosa_con_funcion_no_se_parte():
    assert spanish_keys("por causa de") == ["por causa de"]
    assert spanish_keys("elaborar fabricar") == ["elaborar fabricar", "elaborar", "fabricar"]


def test_la_medida_separa_diccionario_y_modelo():
    assert character_score("anaa", "anaa") == 1
    assert outcome(["aainjaa"], "aainjaa", "otra")["correct"] is True
    missed = outcome(None, "aainjaa", "anaa")
    assert missed["correct"] is False
    assert missed["model_exact"] is False
    assert outcome(None, "aainjaa", "aainjaa")["correct"] is True


def test_hacer_sale_del_diccionario():
    response = client.post(
        "/translate",
        json={"text": "hacer", "source": "es", "target": "guc"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["origin"] == "dictionary"
    assert body["approximate"] is False
    assert "aainjaa" in body["parts"][0]["matches"]


def test_una_forma_desconocida_se_marca_como_aproximacion():
    response = client.post(
        "/translate",
        json={"text": "zzzqqq", "source": "es", "target": "guc"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["approximate"] is True
    assert body["origin"] == "model"


def test_embera_no_inventa_una_traduccion():
    response = client.post(
        "/translate",
        json={"text": "agua", "source": "es", "target": "embera"},
    )
    assert response.status_code == 422
    assert "corpus" in response.json()["error"]


def test_el_texto_vacio_se_rechaza():
    response = client.post(
        "/translate",
        json={"text": "   ", "source": "es", "target": "guc"},
    )
    assert response.status_code == 400


def test_salud_e_idiomas():
    health = client.get("/health")
    languages = client.get("/languages")
    assert health.status_code == 200
    assert health.json()["pairs"] == sum(1 for _ in csv.DictReader(PAIRS.open(encoding="utf-8")))
    ready = {item["code"]: item["ready"] for item in languages.json()["languages"]}
    assert ready == {"es": True, "guc": True, "embera": False}
