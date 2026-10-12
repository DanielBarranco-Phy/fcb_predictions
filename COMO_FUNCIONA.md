# Cómo funciona la predicción

Cuando ejecutas `predict.py` pasan cinco cosas, siempre en este orden. Cada una vive en un archivo de `src/`.

```
datos  →  Elo  →  factores  →  modelo  →  porcentajes
data.py   elo.py  features.py  model.py   predict.py
```

No hay nada guardado de antemano: cada vez que ejecutas, se recalcula todo desde los datos (tarda unos segundos). Por eso basta con actualizar los datos para que la predicción cambie.

---

## 1. Datos (`src/data.py`)

Junta todos los partidos en una sola tabla, un partido por fila: fecha, competición, local, visitante, goles.

| Fuente | Qué aporta | Partidos |
|---|---|---|
| football-data.co.uk | 11 ligas europeas desde la 22/23 (La Liga, Premier, Bundesliga, Serie A, Ligue 1, Portugal, Países Bajos, Bélgica, Turquía, Grecia, Escocia) | ~14.500 |
| fixturedownload.com | Champions League desde la 22/23, de todos los equipos | 646 |
| ESPN (API pública, no oficial) | Copa del Rey y Supercopa del Barça desde la 22/23 | 27 |

**Por qué 11 ligas si la cuenta es del Barça:** las otras ligas no entrenan el modelo. Solo sirven para saber qué tan fuerte es un rival europeo (paso 2).

**Por qué marcador a los 90 minutos en las copas:** la predicción es victoria / empate / derrota en el tiempo reglamentario, igual que las casas de apuestas. Una final que se gana en la prórroga cuenta como empate. ESPN solo da el marcador final, así que en esos partidos se iguala al marcador más bajo (un 3-2 en la prórroga se guarda como 2-2).

---

## 2. Elo (`src/elo.py`) — qué tan fuerte es cada equipo

El Elo es un número por equipo. No es machine learning, es una fórmula que se aplica partido a partido en orden de fecha.

**Antes del partido** se calcula el resultado esperado, entre 0 y 1:

```
esperado_local = 1 / (1 + 10 ^ ( -(elo_local + 60 - elo_visitante) / 400 ))
```

- Los `60` son la ventaja de jugar en casa (0 si el campo es neutral).
- Dos equipos iguales en campo neutral → 0,50.
- 100 puntos de ventaja → 0,64. 300 puntos → 0,85.

**Después del partido** se compara con lo que pasó (victoria = 1, empate = 0,5, derrota = 0):

```
elo_nuevo = elo_viejo + 20 × margen × (real − esperado)
```

- `20` (K) es la velocidad de reacción. Más alto = el Elo cambia más con cada resultado.
- `margen` premia las goleadas: ×1 por un gol, ×1,5 por dos, ×(11 + goles)/8 por tres o más.
- Lo que sube un equipo lo baja el otro.

**Ejemplo.** Barça (1866) recibe a un equipo de 1500. Esperado = 0,92.
- Si gana 1-0: sube 20 × 1 × (1 − 0,92) = **+1,6**. Casi nada, era lo esperado.
- Si pierde 0-1: baja 20 × 1 × (0 − 0,92) = **−18,4**. Mucho, fue una sorpresa.

Así el Elo se corrige solo: ganar lo que debes ganar no te sube, perderlo te hunde.

### El problema de comparar ligas

Dentro de una liga el Elo es de suma cero, así que el campeón turco acabaría con el mismo Elo que el campeón español. Para arreglarlo, cada liga empieza en un nivel distinto, y ese nivel se calcula con los resultados de Champions:

1. Se corre el Elo con todas las ligas empezando en 1500.
2. Se mira qué ligas rindieron en Champions mejor de lo que su Elo esperaba.
3. Se sube el nivel inicial de esas ligas y se baja el de las otras.
4. Se repite 30 veces hasta que se estabiliza.

Niveles resultantes (La Liga fija en 1500 como referencia): Premier 1600, Bundesliga 1520, Ligue 1 1509, **La Liga 1500**, Serie A 1492, Portugal 1368, Turquía 1248.

Los rivales de Copa de divisiones inferiores empiezan en 1200.

**Valores de K y ventaja de local:** probé K entre 10 y 30 y ventaja entre 40 y 80. Casi no hay diferencia; 20 y 60 están entre los mejores y son valores estándar.

---

## 3. Factores (`src/features.py`) — lo que ve el modelo

Para cada partido se calculan cinco números, siempre como **local menos visitante** y usando solo lo que se sabía antes del partido:

| Factor | Qué es |
|---|---|
| `elo_diff` | Elo del local − Elo del visitante |
| `form_pts_diff` | Diferencia de puntos por partido en los últimos 5 (todas las competiciones) |
| `form_gd_diff` | Diferencia de goles a favor menos en contra por partido, últimos 5 |
| `rest_diff` | Diferencia de días de descanso (tope de 10 días) |
| `is_champions` | 1 si es Champions, 0 si no |

**¿Y la localía?** No es un factor aparte: todos los partidos se miran desde el lado del local, así que el modelo aprende la ventaja de casa directamente (en los datos el local gana el 47,5%, el visitante el 29,2% y hay empate en el 23,2%).

**¿Y el histórico contra el rival?** Se muestra en pantalla para el video, pero no entra al modelo. Con dos o tres partidos por rival no es fiable como dato.

---

## 4. Modelo (`src/model.py`) — aquí está el machine learning

**Qué modelo es:** regresión logística multinomial (scikit-learn). Es de los modelos de ML más simples: aprende un peso para cada factor y con ellos saca tres probabilidades que suman 100%: gana el local, empate, gana el visitante.

**Con qué aprende:** 2.031 partidos de La Liga y Champions desde noviembre de 2022, de todos los equipos. No solo del Barça, porque ~60 partidos por temporada son muy pocos para aprender.

- Se descartan los primeros meses de la 22/23: el Elo arranca desde la media de cada liga y tarda unas jornadas en ser fiable.
- Se descartan los partidos en campo neutral (no hay local real).

**Peso por antigüedad (Xavi vs Flick):**

```
peso = 0.5 ^ (días_de_antigüedad / 730)
```

Un partido de hoy pesa 1; uno de hace dos años, 0,5; uno de hace cuatro, 0,25. El modelo "escucha" más a los partidos recientes. En las pruebas esto cambia poco el resultado, porque el Elo y la forma ya miran lo reciente por construcción.

**Lo que aprendió** (peso de cada factor para "gana el local"; más lejos de cero = más importante):

| Factor | Peso |
|---|---|
| `elo_diff` | **+0,56** |
| `form_gd_diff` | +0,15 |
| `form_pts_diff` | −0,12 |
| `is_champions` | +0,06 |
| `rest_diff` | −0,02 |

Casi toda la predicción sale del Elo. La forma aporta poco porque el Elo ya la incluye en parte: un equipo en racha ya subió de Elo.

---

## 5. Predicción (`predict.py`)

1. Toma el Elo actual y los últimos 5 partidos del Barça y del rival.
2. Calcula los cinco factores para el partido que pediste.
3. El modelo devuelve tres probabilidades.
4. **Casa / visitante:** el Barça se pone como local o como visitante según `LUGAR`.
5. **Campo neutral:** se calcula dos veces, una con cada equipo de local, y se promedia. Así se anula la ventaja de casa.
6. Guarda la predicción en `predictions.csv` para llevar el historial de aciertos.

**Para qué sirve la fecha:** solo para dos cosas: calcular los días de descanso y emparejar luego la predicción con el resultado real en el historial. Si la dejas vacía, en Champions se toma del calendario; en el resto se usa la fecha de hoy.

---

## 6. Qué tan bueno es

Prueba honesta: entrenado con partidos hasta junio de 2025 y evaluado en 655 partidos posteriores que nunca vio (`python src/model.py` lo repite).

| | Acierto |
|---|---|
| Decir siempre "gana el local" | 49,2% |
| **Modelo** (La Liga + Champions) | **54,4%** |
| Modelo, solo La Liga | 51,2% |
| Casas de apuestas, solo La Liga | 54,6% |
| Modelo, solo partidos del Barça | 81,0% |
| Decir siempre "gana el Barça" | 79,3% |

En fútbol nadie pasa mucho del 55%: hay demasiado azar en un partido. Lo valioso del modelo no es decir quién gana sino **con qué probabilidad**.

## 7. Lo que el modelo no sabe

- Lesiones, sanciones y alineaciones.
- Rotaciones (si el entrenador guarda titulares antes de un partido grande).
- xG y calidad de las ocasiones: solo ve goles.
- Si la eliminatoria ya está decidida en la vuelta.
- Casi nunca dará el empate como resultado más probable, aunque sí le asigna su porcentaje.

---

## 8. Ajuste por bajas (`predict.py`)

El modelo no sabe quién juega. Para tenerlo en cuenta hay un ajuste manual: en `predict.py` escribes quién falta y de qué tipo es, y se le restan puntos de Elo a ese equipo antes de predecir.

```python
ABSENT_BARCA = {"Raphinha": "star"}
ABSENT_OPPONENT = {}
```

| Tipo | Puntos de Elo que resta |
|---|---|
| `star` | 40 |
| `starter` | 20 |
| `rotation` | 10 |

Para comparar: jugar en casa vale 60 puntos. Perder a una estrella equivale a dos tercios de esa ventaja.

**Importante:** estos valores los elegí yo, no los aprendió el modelo, porque los datos no traen alineaciones. Por eso `predict.py` muestra siempre las dos predicciones (modelo base y con bajas) y `update.py` lleva el historial de ambas. Si con el tiempo la versión con bajas acierta más, vale la pena hacer la versión completa: entrenar el modelo con los onces iniciales y la nota de cada jugador.
