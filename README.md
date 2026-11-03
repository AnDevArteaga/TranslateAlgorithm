# Traductor de lenguas indígenas

Servicio que traduce entre español y wayuunaiki. La búsqueda principal es un diccionario: si la palabra está, devuelve sus sentidos. Si no está, un modelo de secuencias propone una forma y la respuesta la marca como aproximación. Emberá aparece en la lista de idiomas, pero el servicio responde que no hay corpus y no inventa palabras.

El diccionario sale de [Gaxys/wayuu_spa_dict](https://huggingface.co/datasets/Gaxys/wayuu_spa_dict), 2183 filas de pares wayuunaiki–español. Es un léxico, no un corpus de oraciones. Con ese tamaño el modelo no traduce frases como un sistema comercial: una frase se resuelve palabra por palabra.

## Cómo responde

`POST /translate`

```json
{ "text": "hacer", "source": "es", "target": "guc" }
```

Si `hacer` está en el diccionario, `origin` es `dictionary` y `approximate` es `false`. Los sentidos van en `parts[].matches`. Una forma que no está pasa por el modelo, `origin` es `model` y `approximate` es `true`.

Idiomas: `es`, `guc`, `embera`. Emberá, el mismo idioma en origen y destino, o un texto vacío, no se traducen.

`GET /languages` dice cuáles están listos. `GET /health` confirma que el diccionario cargó.

## Arranque

Hace falta Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train
uvicorn src.api:app --port 8000
pytest
```

`python -m src.train` limpia el parquet, escribe `data/wayuu_pairs.csv` y entrena el modelo en CPU. El repositorio ya incluye el CSV limpio y los artefactos de ese entrenamiento, así que el servicio arranca sin volver a entrenar. El parquet crudo no se versiona.

La limpieza quita etiquetas gramaticales (`v_t`, `n_pos`), notas morfológicas (`-ta`, `-ra`), referencias cruzadas y glosas que en realidad son otra forma wayuunaiki. Una glosa como `por causa de` se indexa completa. Una glosa como `elaborar fabricar` también se indexa por cada palabra.

## Precisión

```bash
python -m src.evaluate
```

Aparta el 10% de los pares y entrena solo con el resto. El modelo guardado en `artifacts/` no sirve para esta cifra: ya vio todas las palabras. El acierto exacto es la parte de la prueba en la que la forma generada coincide con la guardada. La cercanía por caracteres es 100% menos la distancia de edición, dividida por la palabra más larga. El diccionario, sobre los pares que sí guarda, devuelve esa forma.

## Imagen

```bash
docker build -t traductor-algoritmo .
docker run --rm -p 8000:8000 traductor-algoritmo
```

El entrenamiento es un comando de desarrollo. No hay un endpoint público para reentrenar.

## Licencia

MIT. Copyright (c) 2026 Andres Arteaga.

Los pares limpios provienen de Gaxys/wayuu_spa_dict. La ficha del dataset no publica una licencia aparte; el crédito queda en el diccionario y en este archivo.
