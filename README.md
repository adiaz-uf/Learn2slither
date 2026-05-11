# Learn2Slither — Implementación técnica del algoritmo de RL

Agente de Snake entrenado con **Deep Q-Learning (DQN)** sobre TensorFlow/Keras.
Este documento describe cómo funciona el algoritmo paso a paso y cómo lo
implementa el código actual.

El estado se codifica siempre como un **vector de 16 features de distancia
tamaño-invariantes** (4 direcciones × `[d_wall, d_body, d_green, d_red]`,
con cada `d` = `1/distancia`), de modo que la misma arquitectura de red
juega en cualquier tamaño de tablero.

El proyecto ofrece **dos modos de entrenamiento**, seleccionables con la
opción `--training-mode` de la CLI. Sólo se diferencian en el **muestreo de
tamaños del tablero** durante el entrenamiento; la red, el encoder y los
hiperparámetros son idénticos:

- **`single`** (por defecto): cada episodio se juega en `--board-size`
  (10, 14 o 18). Modelo especialista en un tamaño concreto.
- **`multi`** (bonus de portabilidad): cada episodio samplea uniformemente
  uno de `[10, 14, 18]` (33% / 33% / 33%). Un único modelo entrenado
  expuesto a los tres tamaños — cubre el bonus del subject de jugar
  cualquier tamaño con un único modelo.

> Para uso e instalación, ver [USAGE.md](USAGE.md).

---

## 1. Arquitectura general

El código se organiza en cinco archivos:

| Archivo                                          | Responsabilidad                                                                              |
|--------------------------------------------------|----------------------------------------------------------------------------------------------|
| [src/AI_Model/main.py](src/AI_Model/main.py)     | CLI, bucle de entrenamiento/evaluación, sampling de tamaños en `multi`, checkpoints, carga de modelos. |
| [src/AI_Model/Agent.py](src/AI_Model/Agent.py)   | DQN: arquitectura de la red, replay buffer, target network, encoder 16-D `process_view`. |
| [src/AI_Model/Board.py](src/AI_Model/Board.py)   | Lógica del juego, generación de visión cruz, recompensas, colisiones.                        |
| [src/AI_Model/Snake.py](src/AI_Model/Snake.py)   | Estructura de datos del snake (cabeza + cuerpo).                                              |
| [src/UI/game_ui.py](src/UI/game_ui.py)           | Visualización pygame: lobby con selector de tamaños, tablero, game-over, stats.              |

El bucle básico de DQN es:

```
observar s  →  elegir a ~ ε-greedy(Q(s,·))  →  ejecutar a
            →  recibir r y s'
            →  guardar (s, a, r, s')
            →  entrenar Q para que se acerque al target de Bellman
```

El target que persigue la red es:

```
Q*(s,a) = r + γ · max_a' Q*(s',a')
```

con `γ = 0.9646` (Optuna-tuned, [Agent.py](src/AI_Model/Agent.py)).

Las dos secciones siguientes explican los **dos pilares conceptuales** del
algoritmo: la **Q-function** (qué queremos calcular) y la **red neuronal**
(con qué la calculamos). En la sección 4 detallamos el encoder 16-D que
alimenta la red.

---

## 2. La Q-function — qué es, qué hace, para qué sirve

### 2.1 ¿Qué es la Q-function?

La Q-function, escrita formalmente `Q(s, a)`, es una **función de dos
argumentos** que devuelve un número:

- `s` = un **estado** del entorno (en nuestro caso, la visión cruz del snake
  ya codificada — ver sección 4).
- `a` = una **acción** posible desde ese estado (UP, DOWN, LEFT, RIGHT).
- `Q(s, a)` = un escalar que llamamos **Q-value** o **valor de la acción**.

> "Q" viene de **Quality** — *cómo de buena* es la acción `a` cuando estoy en
> el estado `s`.

### 2.2 ¿Qué representa exactamente ese número?

`Q(s, a)` es la **recompensa total futura esperada** si:

1. Estoy en el estado `s`,
2. Ejecuto la acción `a`,
3. A partir de ahí actúo siempre de manera óptima.

Matemáticamente:

```
Q(s, a) = E[ r₀ + γ·r₁ + γ²·r₂ + γ³·r₃ + ... ]
              ↑         ↑                ↑
              |         |                pasos lejanos
              |         siguiente paso
              recompensa de ejecutar a ahora
```

`γ = 0.9646` es el discount factor: las recompensas lejanas valen menos que
las inmediatas. El agente prefiere una manzana ahora a una manzana en 100
pasos.

### 2.3 ¿Para qué sirve la Q-function?

**Para tomar decisiones.** Si conozco `Q(s, a)` para todas las acciones `a`,
mi mejor jugada es `argmax_a Q(s, a)`.

Esto es exactamente lo que hace
[Agent.get_action()](src/AI_Model/Agent.py#L221) cuando no está explorando:

```python
state = self.process_view(view)                  # codifica el estado a 16-D
q_values = self.model(state, training=False)     # forward pass
return int(np.argmax(q_values[0]))               # elige la acción
                                                 # de mayor Q-value
```

**La política óptima sale gratis** una vez tienes una buena Q-function. Como
la red emite un vector de 4 Q-values, sólo necesitamos un `argmax` para
decidir.

### 2.4 La ecuación de Bellman — cómo se aprende Q

No conocemos `Q*(s, a)` a priori; debemos aprenderla. La clave es la
**propiedad recursiva de Bellman**:

```
Q*(s, a) = r + γ · max_a' Q*(s', a')
```

Esta ecuación dice: *"el valor de hacer `a` ahora = recompensa inmediata + lo
mejor que pueda hacer después"*. Convierte el problema en aprendizaje
supervisado:

- **Predicción**: `Q(s, a)` (lo que dice la red ahora).
- **Etiqueta (target)**: `r + γ · max_a' Q(s', a')`.
- **Error**: la diferencia — eso es lo que minimizamos.

Cada paso de juego nos da una transición `(s, a, r, s')` que se guarda en el
replay buffer y, más tarde, sirve para construir un target.

### 2.5 ¿Por qué necesitamos aproximarla con una red neuronal?

En problemas pequeños, `Q` cabe en una **Q-table**. Aquí no: el estado es un
vector continuo de 16 floats (4 direcciones × 4 distancias normalizadas en
`[0, 1]`). El cardinal del espacio de estados es infinito — imposible de
tabular.

Aproximamos `Q(s, a)` con una **función paramétrica** `Q(s, a; θ)` que
generaliza entre estados parecidos. Esa función paramétrica es la red
neuronal.

---

## 3. La red neuronal — qué es, qué hace, para qué sirve aquí

### 3.1 ¿Qué hace exactamente la red en este proyecto?

Aproxima la Q-function:

```
Entrada:  vector de 16 floats (4 dirs × [d_wall, d_body, d_green, d_red])
Salida:   vector de 4 floats = [Q(s,UP), Q(s,DOWN), Q(s,LEFT), Q(s,RIGHT)]
```

**Una sola pasada hacia delante** (forward pass) nos da los Q-values de las
cuatro acciones a la vez.

### 3.2 Arquitectura — capa por capa

Definida en [Agent._build_model()](src/AI_Model/Agent.py). El default es un
MLP compacto adecuado al input 16-D (Optuna tuning puede sobrescribir
forma y dropout):

```
Input(16)
    │
    ▼
Dense(64, ReLU)
    │
    ▼
Dense(64, ReLU)
    │
    ▼
Dense(4, lineal)
                                  ─────────────────
                                  ≈ 5 444 parámetros
```

#### Por qué tan pequeña

El input ya contiene la información destilada (¿a qué distancia hay pared?
¿a qué distancia hay manzana?). La red no necesita inferir relaciones
posicionales a partir de un one-hot crudo — basta con combinar 16 distancias.
Pocas neuronas, pocos parámetros, generalización rápida y poco riesgo de
overfitting.

### 3.3 Por qué cada elección de diseño

#### ReLU en capas ocultas

`ReLU(x) = max(0, x)`. Aporta no-linealidad (sin ella, apilar capas sería
equivalente a una sola capa lineal), no satura para valores grandes (evita
*vanishing gradient*) y es computacionalmente barata.

#### Salida lineal

Los Q-values son recompensas acumuladas descontadas, sin acotación natural.
Una sigmoid o softmax los distorsionaría. Aunque la salida tenga 4 neuronas
**esto no es clasificación**: son cuatro escalares de regresión (uno por
acción), por eso la loss es Huber y no cross-entropy.

#### Dropout off por defecto

La red default de ~5 K parámetros es lo suficientemente pequeña como para no
necesitar dropout — añadirlo reduciría la capacidad útil por debajo del
mínimo. Optuna puede activarlo si encuentra una arquitectura más grande
donde compense.

#### Huber loss (δ = 1.0)

[Agent.py:74](src/AI_Model/Agent.py#L74): cuadrática para errores pequeños
(gradientes suaves cerca del óptimo) y lineal para errores grandes
(gradientes acotados). En DQN al inicio los Q-values están descalibrados;
con MSE, errores de magnitud 250 producirían gradientes proporcionales a
62500 → desestabilizan la red.

#### Adam con `lr = 0.000216` y `clipnorm = 1.0`

- **Adam**: combina momentum y learning rate adaptativo por parámetro.
- **Learning rate 0.000216** (Optuna-tuned): bajo a propósito; RL es propenso
  a olvido catastrófico.
- **clipnorm = 1.0**: si la norma del gradiente excede 1, se reescala. Red de
  seguridad complementaria a Huber.

### 3.4 Cómo se entrena la red — `_train_step`

[Agent._train_step()](src/AI_Model/Agent.py#L239), decorado con
`@tf.function` para compilar el grafo:

```python
@tf.function
def _train_step(self, states, actions, targets):
    with tf.GradientTape() as tape:
        all_q = self.model(states, training=True)              # 1. Forward
        masks = tf.one_hot(actions, self.output_size)           # 2. Máscara
        current_q = tf.reduce_sum(all_q * masks, axis=1)        #    aísla
                                                                #    Q(s,a_taken)
        loss = self.loss_fn(targets, current_q)                 # 3. Huber
    grads = tape.gradient(loss, self.model.trainable_variables) # 4. Backprop
    self.optimizer.apply_gradients(zip(grads, ...))             # 5. Adam +
                                                                #    clipnorm
    return loss
```

La red sólo recibe gradiente para la acción que efectivamente se ejecutó (las
otras tres están enmascaradas). **Esta función es idéntica para los dos
modos**; sólo cambia la forma del tensor `states`.

---

## 4. Espacio de estados — la "visión cruz" y sus dos codificaciones

El agente **no ve el tablero completo**. Recibe sólo cuatro líneas de visión
(arriba, abajo, izquierda, derecha) que parten de la cabeza, hasta encontrar
una pared.

[Board.get_snake_view()](src/AI_Model/Board.py) devuelve cuatro listas
ordenadas de la celda más cercana a la más lejana. Cada celda contiene uno
de los caracteres `0` (vacío), `W` (pared), `S` (cuerpo), `G` (manzana
verde) o `R` (manzana roja).

[Agent.process_view()](src/AI_Model/Agent.py) **convierte la visión cruda
en un tensor de 16 floats** que alimenta a la red.

### 4.1 Encoder 16-D — features tamaño-invariantes

[Agent._process_view_features()](src/AI_Model/Agent.py) genera un vector de
**16 dimensiones que NO depende del tamaño del tablero**.

Para cada una de las 4 direcciones emite 4 números:

```
[d_wall, d_body, d_green, d_red]
```

donde cada `d_*` es `1/distancia` a la **primera ocurrencia** de esa entidad
en el brazo:

- `1.0` si la entidad está pegada a la cabeza (distancia 1).
- `0.5` a 2 celdas, `0.333` a 3, `0.1` a 10.
- `0.0` si la entidad **no es visible** en ese brazo.

Total: `4 direcciones × 4 entidades = 16 features`.

### 4.2 Por qué `1/distancia` y no `distancia` cruda

1. **Acota el rango** a `[0, 1]` independientemente del tamaño de tablero.
2. **Da más peso a lo cercano** — la diferencia entre "pared a 1" y "pared a
   2" (1.0 → 0.5) es enorme; entre "pared a 9" y "pared a 10" (0.111 →
   0.100) es despreciable. Refleja la urgencia táctica.
3. **`0.0` representa naturalmente "no visible"**.

### 4.3 Implicaciones del encoder

| Propiedad                                       | Consecuencia                                          |
|-------------------------------------------------|-------------------------------------------------------|
| Input shape fijo (16) en cualquier tamaño       | La misma red juega 10/14/18 — cubre el bonus          |
| Inductive bias en `1/d`                         | Lo cercano pesa más automáticamente                   |
| Solo distancia a la primera entidad por brazo   | No distingue "varias manzanas en línea" (limitación)  |
| Sin información posicional absoluta             | La política aprende relaciones, no posiciones         |

---

## 5. Espacio de acciones

Cuatro acciones discretas: `UP=0`, `DOWN=1`, `LEFT=2`, `RIGHT=3`
([Agent.py:57](src/AI_Model/Agent.py#L57)).

La red produce un Q-value por acción → `output_size = 4`. **Idéntico para
los dos modos.**

En el bucle, [main.py:154](src/AI_Model/main.py#L154) define
`directions_map = ['UP', 'DOWN', 'LEFT', 'RIGHT']`, que traduce el índice
devuelto por `argmax` al string que espera
[Board.move_snake()](src/AI_Model/Board.py#L167).

---

## 6. Función de recompensa

Constantes en [Board.py:13-16](src/AI_Model/Board.py#L13):

| Evento                   | Recompensa            | Constante           | Razón                              |
|--------------------------|-----------------------|---------------------|------------------------------------|
| Manzana verde            | `+50`                 | `GREEN_APPLE`       | Señal positiva fuerte              |
| Manzana roja             | `-10`                 | `RED_APPLE`         | Penalización moderada (encoge)     |
| Movimiento sin comer     | `-1.284`              | `NO_EAT`            | Urgencia por buscar comida         |
| Colisión / loop / suicidio (red apple en cuerpo vacío) | `-100` | `INSTANT_GAMEOVER` | Señal de muerte fuerte             |
| Timeout `max_steps`      | `-50`                 | (literal en main.py) | Episodio demasiado largo           |

La función de recompensa es **idéntica para los dos modos** — la única
diferencia entre `single` y `multi` está en el muestreo de tamaños de
tablero, no en cómo se interpretan eventos.

### Detección de loops — escala con la longitud

[main.py:269-273](src/AI_Model/main.py#L269): el umbral
`max_steps_without_food` se recalcula cada paso a partir de la longitud
actual del snake:

```python
current_length = len(board.snake.body) + 1
max_steps_without_food = (
    current_board_size * current_board_size
    * max(2, current_length // 6)
)
```

Esto da:

| Longitud del snake | `max_steps_without_food` (en 10×10) |
|--------------------|--------------------------------------|
| ≤ 12               | 200                                  |
| 18                 | 300                                  |
| 24                 | 400                                  |
| 30                 | 500                                  |

Cuando se cumple, se marca `done=True` y se aplica `reward = INSTANT_GAMEOVER`
([main.py:275-279](src/AI_Model/main.py#L275)). Snakes largos necesitan más
pasos para esquivar su propio cuerpo.

`max_steps` global ([main.py:218](src/AI_Model/main.py#L218)) es
`current_board_size² × 20` (2000 en 10×10) y termina el episodio con
`reward = -50`.

---

## 7. Política ε-greedy

[Agent.get_action()](src/AI_Model/Agent.py):

```python
if np.random.rand() < epsilon:
    return np.random.randint(0, self.output_size)   # exploración
state = self.process_view(view)
q_values = self.model(state, training=False)
return int(np.argmax(q_values[0]))                  # explotación
```

### Schedule de ε en tres fases

[main.py:189-192, 401-423](src/AI_Model/main.py#L189):

```python
warmup_episodes = 200
decay_end_episode = 200 + min(int((num_episodes - 200) * 0.35), 4500)
epsilon_target = 0.01
```

1. **Warmup** (primeros 200 episodios): `ε = initial_epsilon` (1.0 desde
   cero, 0.005 si se carga un modelo) — exploración pura para llenar el
   buffer.
2. **Decaimiento lineal**: hasta `decay_end_episode`, descenso lineal de
   `initial_epsilon → 0.01`. El `35%` se cap-ea en 4500 para que runs largos
   (ej. 20 K episodios) no se pasen mucho tiempo en pura exploración.
3. **Explotación con ciclos**: a partir de `decay_end_episode`, en bloques de
   200 episodios, los **primeros 30** suben a `ε = 0.05` **y** elevan el
   learning rate a `5 × lr_base`; los siguientes 170 vuelven a `ε = 0.01` y
   `lr = lr_base`. El spike de lr permite que las transiciones nuevas
   obtenidas mediante exploración tengan suficiente peso sobre los 150 K de
   buffer mayoritariamente greedy.

El learning rate se actualiza en caliente con `tf.Variable.assign()`
([Agent.set_learning_rate()](src/AI_Model/Agent.py#L124)) — sin retracear el
grafo `@tf.function`.

El schedule es **el mismo para los dos modos**. En `multi`, además, el
`board_size` cambia de un episodio al siguiente, pero ε no se ve afectada.

---

## 8. Experience Replay

Buffer en [Agent.__init__](src/AI_Model/Agent.py#L81):

```python
self.memory = deque(maxlen=150000)
self.batch_size = 512
```

Cada paso de juego:

- [Agent.remember()](src/AI_Model/Agent.py#L235) guarda la transición cruda
  `(state_view, action, reward, next_state_view, done)`.
- [Agent.replay()](src/AI_Model/Agent.py#L253) muestrea **aleatoriamente** un
  mini-batch de 512 transiciones y entrena.

¿Por qué muestrear aleatoriamente? Las transiciones consecutivas están
**altamente correlacionadas** (si el agente camina recto durante 20 pasos,
esos 20 ejemplos son casi idénticos). Entrenar con ellos en orden hace que la
red sobre-ajuste el comportamiento reciente. El sampling rompe la
correlación.

**Capacidad 150 K**: a ~200 pasos/episodio cubre ≈750 episodios — suficiente
para no olvidar lo aprendido en el warmup cuando el agente entra en
explotación.

> En el modo `multi`, el buffer mezcla transiciones de los tres tamaños
> (sampleadas uniformemente por episodio). Cada mini-batch tiene una
> distribución representativa, lo que **fuerza a la red a aprender una
> política que funcione en los tres tableros simultáneamente**.

> En el buffer se guardan **vistas crudas** (las listas de chars que
> devuelve `Board.get_snake_view()`), no los tensores codificados. La
> codificación se aplica al sacar las muestras del buffer.

---

## 9. Target Network + Double DQN

[Agent.__init__](src/AI_Model/Agent.py#L89):

```python
self.model = self._build_model()         # red principal (entrenada cada paso)
self.target_model = self._build_model()  # red objetivo (sincronizada cada 1000 pasos)
self.update_target_model()
```

### Problema que resuelve

Si usamos la **misma red** para calcular `Q(s,a)` (predicción) y `Q(s',a')`
(target), tenemos un objetivo móvil: actualizar la red cambia simultáneamente
predicción y target → inestabilidad y oscilaciones.

**Target network**: copiamos los pesos cada 1000 pasos
([target_update_frequency](src/AI_Model/Agent.py#L85)). Entre copias los
targets están "congelados" y la red persigue un objetivo estable.

### Double DQN

DQN estándar sufre **sesgo de sobreestimación**: aplicar `max` sobre
estimaciones ruidosas siempre infla el valor.

[Agent.replay()](src/AI_Model/Agent.py#L289) implementa Double DQN
(Hasselt 2015):

```python
# Red principal SELECCIONA la mejor acción
next_q_main = self.model(next_states, training=False).numpy()
best_actions = np.argmax(next_q_main, axis=1)

# Red target EVALÚA esa acción
next_q_target = self.target_model(next_states, training=False).numpy()
max_next_q = next_q_target[np.arange(len(best_actions)), best_actions]

targets = rewards + (1 - dones) * self.gamma * max_next_q
```

Separar las dos roles elimina el sesgo: si la red principal sobreestima una
acción, es improbable que la target también lo haga.

---

## 10. Anatomía del flujo de entrenamiento — paso a paso

### 10.1 Vista de pájaro

[main.py train_agent()](src/AI_Model/main.py#L113):

```
para cada episodio (1 a num_episodes):
    [si modo multi] elegir current_board_size ← _sample_board_size()
    crear Board(current_board_size + 2) e inicializar serpiente

    mientras no done:
        ─────────── 1 PASO DE JUEGO ───────────
        s     = board.get_snake_view()                  ← Paso 1
        a     = agent.get_action(s, epsilon)            ← Paso 2
        s', r = board.move_snake(directions_map[a])     ← Paso 3
        actualizar contador anti-loop                   ← Paso 4a
        chequear done (collision / loop / max_steps)    ← Paso 4b
        agent.remember(s, a, r, s', done)               ← Paso 5
        agent.replay()                                  ← Paso 6  ← corazón DQN
        ────────────────────────────────────────

    actualizar ε según schedule (sección 7)
    guardar checkpoints best_*/best_avg_* si procede
```

> El selector de `board_size` por episodio sólo existe en `multi`. En
> `single` el tablero queda fijado al arranque por `--board-size`.

### 10.2 Paso 1 — observar el estado

[main.py:237](src/AI_Model/main.py#L237):

```python
current_view = board.get_snake_view()
```

`get_snake_view()` ([Board.py:121](src/AI_Model/Board.py#L121)) devuelve
4 listas `[up, down, left, right]` con chars `0/W/S/G/R/H`. La salida es
**igual para los dos modos**; la codificación a tensor ocurre dentro del
agente.

### 10.3 Paso 2 — decidir acción (ε-greedy)

[main.py:240](src/AI_Model/main.py#L240) → [Agent.py:221](src/AI_Model/Agent.py#L221):

```python
action_idx = agent.get_action(current_view, epsilon)
```

Internamente `get_action` llama a `process_view` para obtener el tensor 16-D,
hace un forward pass y devuelve el `argmax`.

### 10.4 Paso 3 — ejecutar la acción

[main.py:241,244](src/AI_Model/main.py#L241) → [Board.py:167](src/AI_Model/Board.py#L167):

```python
move_str = directions_map[action_idx]
new_head, reward = board.move_snake(move_str)
```

`move_snake` aplica las reglas: detecta colisión (pared, cuerpo propio),
gestiona comer manzana (`G` crece, `R` encoge — y si el cuerpo está vacío,
encogerse mata) y devuelve la recompensa primaria. `new_head is None` indica
colisión.

### 10.5 Paso 4 — chequeo de fin de episodio

[main.py:248-285](src/AI_Model/main.py#L248):

1. Actualizar `steps_since_food` (cero si el score cambió).
2. `done = (new_head is None)` por colisión.
3. Si `steps_since_food >= max_steps_without_food` (calculado en función de
   la longitud actual, ver sección 6) → `done = True`,
   `reward = INSTANT_GAMEOVER`.
4. Si `step_count >= max_steps` → `done = True`, `reward = -50`.

### 10.6 Paso 5 — guardar la transición

[main.py:287,290](src/AI_Model/main.py#L287):

```python
next_view = board.get_snake_view() if not done else current_view
agent.remember(current_view, action_idx, reward, next_view, done)
```

Se guarda la **vista cruda** (no el tensor). Si el episodio terminó, la
`next_view` es irrelevante; usamos la actual como placeholder.

### 10.7 Paso 6 — `replay()` — corazón de DQN

[main.py:291](src/AI_Model/main.py#L291) →
[Agent.py:253](src/AI_Model/Agent.py#L253). Aquí convergen **todos los
trucos**: experience replay, target network, Double DQN, Huber, Adam, sync.

#### 6a — Sample aleatorio del buffer

```python
if len(self.memory) < self.batch_size:
    return None
minibatch = random.sample(self.memory, self.batch_size)
```

Si aún no hay 512 transiciones, no hace nada (los primeros pasos del warmup
sólo llenan el buffer). Después, sample uniforme.

#### 6b — Codificar el batch

[Agent.py:267-279](src/AI_Model/Agent.py#L267):

```python
states, next_states, actions, rewards, dones = [], [], [], [], []
for sv, a, r, nsv, d in minibatch:
    states.append(self.process_view(sv)[0])
    next_states.append(self.process_view(nsv)[0])
    actions.append(a); rewards.append(r); dones.append(d)
```

→ Aplica el encoder 16-D `1/distancia` a cada visión y construye arrays
NumPy listos para alimentar a la red.

#### 6c — Construir el target de Bellman (Double DQN + target network)

[Agent.py:289-295](src/AI_Model/Agent.py#L289):

```python
next_q_main   = self.model(next_states, training=False).numpy()
best_actions  = np.argmax(next_q_main, axis=1)               # ① principal SELECCIONA
next_q_target = self.target_model(next_states, training=False).numpy()
max_next_q    = next_q_target[np.arange(...), best_actions]  # ② target EVALÚA
targets       = rewards + (1 - dones) * self.gamma * max_next_q
```

Tres piezas a la vez: **ecuación de Bellman**, **target network** (red
estable que aporta `Q(s', ·)`), **Double DQN** (separa selección y
evaluación).

#### 6d — Paso de gradiente (`_train_step`)

[Agent.py:297](src/AI_Model/Agent.py#L297) llama a
[`_train_step`](src/AI_Model/Agent.py#L239) (sección 3.4). **Esta es la
única operación de todo el flujo que modifica los pesos `θ` de la red.**

#### 6e — Sync de la target network cada 1000 pasos

[Agent.py:301-307](src/AI_Model/Agent.py#L301):

```python
self.train_count += 1
if self.train_count % self.target_update_frequency == 0:
    self.update_target_model()
```

`update_target_model()` ([Agent.py:120](src/AI_Model/Agent.py#L120)) copia
`self.model.get_weights() → self.target_model`.

### 10.8 Tabla resumen del flujo

| #  | Acción                            | Código                                                         | Pieza DQN                                       |
|----|-----------------------------------|----------------------------------------------------------------|-------------------------------------------------|
| 1  | Observar estado `s`               | [Board.py:121](src/AI_Model/Board.py#L121)                     | Estado / observación                            |
| 2  | Decidir acción `a`                | [Agent.py:221](src/AI_Model/Agent.py#L221)                     | Política ε-greedy                               |
| 3  | Ejecutar `a` → recibir `r, s'`    | [Board.py:167](src/AI_Model/Board.py#L167)                     | Interacción agente ↔ entorno                    |
| 4  | Chequeo done                      | [main.py:256-285](src/AI_Model/main.py#L256)                   | Terminación de episodio                         |
| 5  | Guardar transición                | [Agent.py:235](src/AI_Model/Agent.py#L235)                     | Experience Replay (escritura)                   |
| 6a | Sample aleatorio del buffer       | [Agent.py:262-265](src/AI_Model/Agent.py#L262)                 | Experience Replay (lectura)                     |
| 6b | Encoding del batch                | [Agent.py](src/AI_Model/Agent.py) `process_view`               | Encoder 16-D                                    |
| 6c | Target de Bellman                 | [Agent.py:289-295](src/AI_Model/Agent.py#L289)                 | Bellman + Target net + Double DQN               |
| 6d | Forward + Huber + backprop + Adam | [Agent.py:239-251](src/AI_Model/Agent.py#L239)                 | **Update de los pesos `θ`**                     |
| 6e | Sync target network               | [Agent.py:301-307](src/AI_Model/Agent.py#L301)                 | Target network update                           |

### 10.9 Final de cada episodio

[main.py:298-394](src/AI_Model/main.py#L298): cuando `done = True`:

1. **Resumen impreso**: pasos, score final, recompensa total, avg/step.
   Detallado para los primeros 20 episodios o si `--debug`; resumido en una
   sola línea para el resto.
2. **Update de stats**: `total_episodes`, `total_score`, `scores`.
3. **Ruta de guardado** (decidida una sola vez por episodio):
   - Modo `multi`: `model_dir = "models/multi"`, `name_tag = "multi"`.
   - Modo `single`: `model_dir = f"models/{board_size}x{board_size}"`,
     `name_tag = f"{board_size}x{board_size}"`.
4. **Doble checkpoint**:
   - Si `final_score > stats['high_score']` → guarda en
     `{model_dir}/best_snake_{name_tag}_{num_episodes}ep.keras`. Es el high
     score individual; puede ser un pico afortunado.
   - Si la **media móvil de 200 episodios** mejora `stats['best_rolling_avg']`
     → guarda en `{model_dir}/best_avg_snake_{name_tag}_{num_episodes}ep.keras`.
     Refleja política consistentemente buena, no un golpe de suerte.
5. Cada checkpoint escribe junto al `.keras` un `*_metadata.json` con `mode`,
   `board_size` (None en multi), `total_episodes`, `high_score`, etc.
6. **Update de ε** según el schedule de tres fases (sección 7).

Tras el último episodio, [main.py:865-879](src/AI_Model/main.py#L865) guarda
el **modelo final** en `models/multi/snake_multi_{N}ep_{timestamp}.keras` o
`models/{S}x{S}/snake_{S}x{S}_{N}ep_{timestamp}.keras`, también con
metadata.

### 10.10 Mapa mental: dónde entra cada concepto

```
┌─────────────────────────────────────────────────────────────────┐
│                   BUCLE DE ENTRENAMIENTO                        │
│                                                                 │
│  Paso 1  ←  Estado          (visión cruz, lista de chars)       │
│  Paso 2  ←  ε-greedy        (process_view → red predictiva)     │
│  Paso 3  ←  Recompensa r    (creada por el entorno)             │
│  Paso 4  ←  Detección de done (collision / loop / timeout)      │
│  Paso 5  ←  Replay buffer   (vistas crudas, no tensores)        │
│  Paso 6a ←  Replay buffer   (sampling aleatorio)                │
│  Paso 6b ←  process_view sobre el batch (encoder 16-D)          │
│  Paso 6c ←  Bellman + Target network + Double DQN               │
│            (← AQUÍ las recompensas se transforman en targets)   │
│  Paso 6d ←  Backprop + Huber + Adam                             │
│            (← AQUÍ los pesos θ de la red cambian)               │
│  Paso 6e ←  Target network sync                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

Las recompensas viven en el buffer durante muchos pasos antes de "tocar" la
red. Cuando lo hacen, lo hacen indirectamente: forman el `target` que la red
persigue.

---

## 11. Modo evaluación (`--dontlearn`)

[main.py:454](src/AI_Model/main.py#L454) (`evaluate_agent`) y
[main.py:566](src/AI_Model/main.py#L566) (`run_with_ui`) son las dos rutas de
evaluación. En ambas, el agente juega con `epsilon=0` (explotación pura) y
**nunca** se llama a `replay()` — la red no se modifica.

### 11.1 Sin UI (`--visual off`, default)

`evaluate_agent` corre `--sessions` partidas seguidas. En cada paso imprime
en terminal la visión cruz (`print_snake_vision`,
[main.py:40](src/AI_Model/main.py#L40)) y la acción elegida. Si
`--step-by-step`, espera Enter entre pasos.

```bash
python src/AI_Model/main.py --dontlearn --sessions 10 \
    --load models/10x10/best_snake_10x10_9000ep.keras
```

### 11.2 Con UI (`--visual on`)

`run_with_ui` lanza una sesión interactiva con tres pantallas pygame:

1. **`StartScreen`** ([game_ui.py:26](src/UI/game_ui.py#L26)) — lobby con
   selector de tamaño. La lista de tamaños depende del modelo cargado:
   - **Modelos `single`**:
     `allowed_sizes = [agent.board_size]` → un solo botón visible.
   - **Modelos `multi`**:
     `allowed_sizes = StartScreen.BOARD_SIZES = [10, 14, 18]` → tres botones.

   Cuando los `allowed_sizes` son menos que los por defecto, la pantalla
   muestra además la pista *"(this model only supports the sizes shown)"*.

2. **`BoardGameUI`** ([game_ui.py:252](src/UI/game_ui.py#L252)) — ventana
   con el tablero en cuadrícula. El agente juega; en cada paso se imprime
   también la visión y la acción en terminal.

3. **`GameOverScreen`** ([game_ui.py:128](src/UI/game_ui.py#L128)) — al
   morir, muestra el score final y dos botones: *New Game* (vuelve a jugar
   con el mismo tamaño) y *View Stats* (`StatsScreen`,
   [game_ui.py:188](src/UI/game_ui.py#L188), con totals/high/avg).

```bash
python src/AI_Model/main.py --dontlearn --visual on \
    --load models/multi/best_snake_multi_10000ep.keras
```

---

## 12. CLI completa

Definida en [main.py](src/AI_Model/main.py):

| Argumento            | Default      | Descripción                                                       |
|----------------------|--------------|-------------------------------------------------------------------|
| `--sessions`         | `100`        | Episodios de entrenamiento o partidas de evaluación.              |
| `--load`             | `None`       | Path a `.keras` para cargar (lee `*_metadata.json` si existe).    |
| `--save`             | `None`       | Path para el modelo final; si se omite se autogenera con timestamp. |
| `--dontlearn`        | `False`      | Modo evaluación (no llama a `replay()`).                          |
| `--visual`           | `off`        | `on` activa la UI pygame (lobby + tablero).                       |
| `--step-by-step`     | `False`      | Pausa cada paso (sólo evaluación).                                |
| `--board-size`       | `10`         | Choices `[10, 14, 18]`. En `multi`: sólo display fallback.        |
| `--training-mode`    | `single`     | Choices `single` / `multi`.                                       |
| `--fps`              | `10`         | FPS de la visualización pygame.                                   |
| `--debug`            | `False`      | Output detallado en los 2 primeros episodios.                     |
| `--learning-rate`    | `0.000216`   | Adam base learning rate.                                          |
| `--gamma`            | `0.9646`     | Discount factor.                                                  |
| `--batch-size`       | `512`        | Mini-batch del replay buffer.                                     |
| `--no-eat-reward`    | `None`       | Override de `Board.NO_EAT` (default `-1.284`).                    |
| `--hidden-layers`    | `None`       | Capas ocultas, e.g. `"64,64"` o `"256,128,64"`. Default: `64,64`. |
| `--dropout`          | `None`       | Dropout entre capas ocultas. Default: `0.0`.                      |

> Modelos guardados con el encoder one-hot legacy (`input_shape != 16`) se
> rechazan al cargar con un mensaje de error claro — hay que reentrenar.

---

## 12bis. Cookbook — comandos para todo

> Todos los comandos asumen el venv ya creado (`make setup`). Las
> alternativas con `python …` son equivalentes a las recetas con `make`
> pero permiten flags fuera del Makefile.

### 12bis.1 Setup inicial

```bash
make setup        # crea .venv y pip install -r requirements.txt
```

### 12bis.2 Entrenamiento — single mode (especialista en un tamaño)

```bash
# 10K episodios en 10×10, sin visualización
make train SESSIONS=10000 BOARD=10

# Idem 14×14 y 18×18
make train SESSIONS=10000 BOARD=14
make train SESSIONS=10000 BOARD=18

# Con UI pygame (mucho más lento, sólo para inspección visual)
make train-visual SESSIONS=200 BOARD=10 FPS=30

# Debug verbose en los 2 primeros episodios
make train-debug SESSIONS=50 BOARD=10

# Continuar entrenamiento desde un checkpoint
make train-continue SESSIONS=5000 BOARD=10 \
    MODEL=models/10x10/best_avg_snake_10x10_10000ep.keras

# Con hiperparámetros custom (sobreescribe los defaults Optuna-tuned)
python src/AI_Model/main.py --sessions 10000 --board-size 10 \
    --learning-rate 0.0005 --gamma 0.97 --hidden-layers 128,64
```

### 12bis.3 Entrenamiento — multi mode (33/33/33, bonus)

```bash
# 10K episodios sampleando uniformemente 10/14/18
make train-multi SESSIONS=10000

# Con debug
make train-multi-debug SESSIONS=50

# Continuar multi desde un checkpoint
make train-multi-continue SESSIONS=5000 \
    MODEL=models/multi/best_avg_snake_multi_10000ep.keras
```

### 12bis.4 Evaluación

```bash
# Terminal: 30 partidas en 10×10 (default board)
make eval SESSIONS=30 BOARD=10 \
    MODEL=models/10x10/best_avg_snake_10x10_10000ep.keras

# UI pygame: lobby con selector de tamaño + tablero animado
make eval-visual BOARD=10 \
    MODEL=models/10x10/best_avg_snake_10x10_10000ep.keras

# UI con selector de los 3 tamaños (modelo multi)
make eval-multi-visual \
    MODEL=models/multi/best_avg_snake_multi_10000ep.keras

# Step-by-step en terminal (Enter entre pasos)
make eval-step BOARD=10 \
    MODEL=models/10x10/best_avg_snake_10x10_10000ep.keras

# Step-by-step con UI (cualquier tecla entre pasos)
python src/AI_Model/main.py --dontlearn --visual on --step-by-step \
    --load models/multi/best_avg_snake_multi_10000ep.keras
```

### 12bis.5 Modelos requirement del subject (1, 10, 100 sessions)

```bash
# Genera 1, 10, 100 sessions en /models/<size>x<size>/ para los 3 tamaños
make models             # = models-10 + models-14 + models-18

# Sólo un tamaño concreto
make models-10
make models-14
make models-18

# Modelo multi-size correspondiente (bonus)
make models-multi
```

### 12bis.6 Optuna — fine-tuning de hiperparámetros

**Importante**: single y multi se tunean **por separado** (las tareas son
distintas, los óptimos pueden divergir). Cada modo guarda en su propio
study SQLite.

```bash
# Single 10×10 — recomendado para el modelo "especialista"
make tune TRIALS=30 BOARD=10
# (= 30 trials × 5000 ep, study: snake-single-10)

# Single 14×14 / 18×18
make tune TRIALS=30 BOARD=14
make tune TRIALS=30 BOARD=18

# Multi (33/33/33) — para el bonus
make tune-multi TRIALS=30
# (study: snake-multi)

# Smoke test rápido (5-10 min, 500 ep/trial, sin pruning)
make tune-fast TRIALS=10 BOARD=10
```

**Inspección de resultados** (cuando termine, o en paralelo desde otra
terminal mientras corre — es sólo lectura):

```bash
# Best params + parameter importance + good zone + comando ready-to-paste
make tune-inspect BOARD=10
make tune-inspect-multi
```

**Continuar/extender un study existente**: simplemente vuelve a lanzar
`make tune` con el mismo board/mode. Optuna detecta el SQLite y añade los
nuevos trials encima:

```bash
# Tienes 12 trials de un tune anterior y quieres 20 más:
make tune TRIALS=20 BOARD=10
# Ahora hay 32 trials en el study; TPE usa los 12 anteriores como contexto.
```

**Controlar el pruning** (lenient por default — ver sección 14.2):

```bash
# Sin pruning ninguno (max safety)
make tune TRIALS=30 BOARD=10 PRUNER=none

# Pruning agresivo (Median, sin patience) — ahorra más tiempo, riesgo mayor
make tune TRIALS=30 BOARD=10 PRUNER=median PATIENCE=0

# Más conservador todavía (mata sólo el bottom-10%)
make tune TRIALS=30 BOARD=10 PRUNE_PCT=10

# Pruning desde más tarde (no juzgar antes del episodio 2500)
make tune TRIALS=30 BOARD=10 N_WARMUP_STEPS=2500
```

**Entrenar el modelo final con los hiperparámetros ganadores**: al final
de `tune-inspect` aparece un comando ready-to-paste. Cópialo y lánzalo:

```bash
python src/AI_Model/main.py --sessions 10000 \
    --training-mode single --board-size 10 \
    --learning-rate 0.000307 --gamma 0.9866 --batch-size 128 \
    --no-eat-reward -1.890 --hidden-layers 320,128
```

### 12bis.7 Workflow de defensa del proyecto

Recomendación de orden si quieres maximizar nota:

```bash
# 1. Setup
make setup

# 2. Tune single 10×10 toda la noche (~5h)
make tune TRIALS=30 BOARD=10

# 3. Por la mañana, ver resultados
make tune-inspect BOARD=10

# 4. Entrenar modelo single 10×10 final con los mejores params
#    (usa el comando ready-to-paste del paso anterior, ~3-5h)
python src/AI_Model/main.py --sessions 10000 [...best params...]

# 5. Generar los modelos 1/10/100 que pide el subject
make models

# 6. Tune multi por separado (~5-7h)
make tune-multi TRIALS=30

# 7. Modelo final multi
make tune-inspect-multi
python src/AI_Model/main.py --sessions 10000 [...multi best params...]

# 8. Verificación visual
make eval-visual BOARD=10 MODEL=<single 10x10 final>
make eval-multi-visual MODEL=<multi final>
```

---

## 13. Bonus — portabilidad de tamaño (`--training-mode multi`)

El subject incluye como bonus que el mismo modelo entrenado pueda jugar en
tableros 10×10, 14×14 y 18×18. Esto se cubre con el modo `multi` — un único
modelo, expuesto durante el entrenamiento a los tres tamaños.

### 13.1 Diferencias respecto al modo `single`

Arquitectura, encoder e hiperparámetros son **idénticos**. La única diferencia
es el muestreo del tamaño de tablero por episodio:

| Aspecto                          | `single` (default)               | `multi` (bonus)                          |
|----------------------------------|----------------------------------|------------------------------------------|
| Tamaño de tablero entrenamiento  | Fijo (`--board-size`)            | Sampleado uniformemente: 33% / 33% / 33% |
| Carpeta de salida                | `models/<size>x<size>/`          | `models/multi/`                          |
| `name_tag` en filename           | `<size>x<size>`                  | `multi`                                  |
| Metadata `mode` en JSON          | `"single"`                       | `"multi"`                                |
| Metadata `board_size`            | El número (10/14/18)             | `null`                                   |
| Botones en la UI                 | 1 (sólo el tamaño entrenado)     | 3 (selección libre)                      |

### 13.2 Sampling uniforme 33/33/33

[main.py](src/AI_Model/main.py):

```python
MULTI_BOARD_SIZES = [10, 14, 18]
MULTI_BOARD_WEIGHTS = [1/3, 1/3, 1/3]
```

Cada episodio se elige aleatoriamente uno de los tres tamaños con la misma
probabilidad. La red ve los tres regímenes (10×10 corto-y-denso, 14×14
intermedio, 18×18 largo-y-disperso) por igual a lo largo del entrenamiento.

### 13.3 Por qué la representación 16D es portable

La red aprende a partir de **distancias relativas**, que son invariantes a
la longitud del brazo. Un Q-value alto para "ir UP cuando hay manzana verde
a `1/3` de distancia y pared a `1/8`" tiene el mismo significado en
cualquier tamaño de tablero — la red no aprende posiciones, sino
**relaciones espaciales**.

### 13.4 Flujo de selección de tamaño en runtime (evaluación)

```
                    ┌──────────────────────────┐
   --load model →   │   leer *_metadata.json   │
                    │   → mode, board_size     │
                    └────────────┬─────────────┘
                                 ▼
                    ┌──────────────────────────┐
                    │   Agent(mode=…)           │
                    └────────────┬─────────────┘
                                 ▼
       --visual on ?  ──────────────────────────
              │                                 │
              ▼ sí                              ▼ no
   ┌─────────────────────────┐      ┌─────────────────────────┐
   │ run_with_ui:            │      │ evaluate_agent usa      │
   │ allowed_sizes según mode │      │ --board-size de CLI     │
   │  • single: [bs]         │      │ (en multi, cualquier    │
   │  • multi : [10,14,18]   │      │  tamaño es válido)      │
   │ StartScreen → tamaño    │      └─────────────────────────┘
   └────────────┬────────────┘
                ▼
   tamaño elegido → BoardGameUI(size+2) → run loop
```

---

## 14. Hiperparámetros — resumen

Idénticos para los dos modos (la única diferencia entre `single` y `multi`
es el muestreo de tamaños).

| Parámetro                     | Valor                | Notas                                              |
|-------------------------------|----------------------|----------------------------------------------------|
| Learning rate base            | 0.000216             | Optuna-tuned                                       |
| Learning rate (spike cycles)  | 5 × lr_base          | Spike de exploración cada 200 episodios            |
| Discount factor `γ`           | 0.9646               | Optuna-tuned                                       |
| Replay buffer size            | 150 000              | ~750 episodios de cobertura                        |
| Mini-batch                    | 512                  |                                                    |
| Target update freq            | 1000 steps           |                                                    |
| ε inicial / final             | 1.0 / 0.01           |                                                    |
| ε al continuar                | 0.005                | Cuando se carga un modelo y se continúa            |
| Warmup episodios              | 200                  | Llenado de buffer antes de empezar a entrenar      |
| Decay end                     | 200 + min(35%·N, 4500) | Fin del decaimiento lineal de ε                  |
| Cycle period / spike length   | 200 / 30             | Periodos de exploración en la fase de explotación  |
| Gradient clipping (norm)      | 1.0                  | Adam clipnorm                                      |
| Loss                          | Huber δ = 1          |                                                    |
| Input size                    | 16                   | Encoder `1/distancia` × 4 entidades × 4 direcciones |
| Capas ocultas (default)       | 64, 64               | Optuna puede sobrescribir vía `--hidden-layers`    |
| Dropout (default)             | 0.0                  | Optuna puede activarlo                             |
| Reward `NO_EAT`               | -1.284               | Optuna-tuned                                       |
| Reward `GREEN_APPLE`          | +50                  |                                                    |
| Reward `RED_APPLE`            | -10                  |                                                    |
| Reward `INSTANT_GAMEOVER`     | -100                 |                                                    |
| Reward timeout `max_steps`    | -50                  | Penalización de paso por exceso                    |
| Pesos sampling tamaños        | 1/3, 1/3, 1/3        | Sólo en `multi`                                    |
| `max_steps` (timeout)         | `N² × 20`            |                                                    |
| `max_steps_without_food`      | `N² × max(2, len/6)` |                                                    |

### 14.1 Hiperparámetros tuneables por Optuna

| Hiperparámetro | Rango / espacio                  |
|----------------|----------------------------------|
| `learning_rate`| log-uniform `[1e-4, 3e-3]`       |
| `gamma`        | uniform `[0.92, 0.99]`           |
| `batch_size`   | categorical `{128, 256, 512, 1024}` |
| `NO_EAT`       | uniform `[-2.0, -0.1]`           |
| `num_layers`   | int `[2, 4]`                     |
| `units_l*`     | int `[32, 256]` step 32 (cada capa) |

Los demás (replay size, gamma del LR spike, target sync, recompensas
distintas a NO_EAT) están fijados como constantes — se pueden cambiar
manualmente vía CLI o editando el código si quieres ampliar el espacio.

### 14.2 Pruning de Optuna — comportamiento por defecto

Para no matar trials buenos por error, el pruning es **lenient**:

| Mecanismo            | Default                     | Función                                        |
|----------------------|-----------------------------|------------------------------------------------|
| Pruner base          | `PercentilePruner(25)`      | Sólo mata el bottom-25% de trials por checkpoint (no el bottom-50% del `MedianPruner` clásico). |
| Wrapper              | `PatientPruner(patience=2)` | Un trial necesita **2 lecturas malas consecutivas** antes de poder morir. |
| `n_warmup_steps`     | `1500`                      | No se considera prunear hasta el episodio 1500 — la fase de decay de ε ha terminado. |
| `n_startup_trials`   | `5`                         | Los primeros 5 trials siempre se completan (poblar la base de comparación). |
| `interval_steps`     | `100`                       | Frecuencia de chequeo (cada 100 episodios reportados). |
| Métrica de comparación | rolling-200 avg de scores | Más estable que el score puntual.              |

Para más / menos agresividad:

```bash
make tune PRUNER=none                # 0 pruning
make tune PRUNER=median PATIENCE=0   # clásico Optuna (agresivo)
make tune PRUNE_PCT=10               # mata sólo el bottom-10%
make tune N_WARMUP_STEPS=2500        # juzgar más tarde aún
```

---

## 15. Estructura de archivos generados

Tras un run de entrenamiento se crean (según el modo):

```
models/
├── 10x10/                                           # single, --board-size 10
│   ├── best_snake_10x10_<N>ep.keras                 # high score individual
│   ├── best_snake_10x10_<N>ep_metadata.json
│   ├── best_avg_snake_10x10_<N>ep.keras             # mejor rolling-200
│   ├── best_avg_snake_10x10_<N>ep_metadata.json
│   ├── snake_10x10_<total_ep>ep_<timestamp>.keras   # final-save
│   └── snake_10x10_<total_ep>ep_<timestamp>_metadata.json
├── 14x14/  ...                                      # single, --board-size 14
├── 18x18/  ...                                      # single, --board-size 18
└── multi/                                           # --training-mode multi
    ├── best_snake_multi_<N>ep.keras
    ├── best_snake_multi_<N>ep_metadata.json
    ├── best_avg_snake_multi_<N>ep.keras
    ├── best_avg_snake_multi_<N>ep_metadata.json
    ├── snake_multi_<total_ep>ep_<timestamp>.keras
    └── snake_multi_<total_ep>ep_<timestamp>_metadata.json
```

Cada `*_metadata.json` contiene como mínimo:

```json
{
  "mode": "single" | "multi",
  "board_size": 10 | 14 | 18 | null,
  "total_episodes": <int>,
  "high_score": <int>,
  "average_score": <float>,            // o "rolling_200_avg" en best_avg_*
  "epsilon" | "final_epsilon": <float>,
  "timestamp": "<ISO8601>"
}
```

`mode` y `board_size` son los campos **clave** para que `--load` reconstruya
la arquitectura correcta del `Agent`.

### 15.1 Artefactos de Optuna

```
optuna_studies/
├── snake-single-10.db    # SQLite con trials, params, valores, estados
├── snake-single-14.db
├── snake-single-18.db
└── snake-multi.db

optuna_results/
└── best_<mode>_<timestamp>.json   # snapshot del best trial al cierre
```

La SQLite es la fuente de verdad — `make tune-inspect` lee directamente de
ahí. El JSON es un snapshot que se reescribe al final de cada `make tune`.

Re-lanzar `make tune` con el mismo `--study-name` (default: derivado de
`mode`+`board`) **continúa el study existente**, añadiendo trials encima.

---

## 16. Referencias

- Mnih et al. *Human-level control through deep reinforcement learning.* Nature, 2015. — DQN base.
- Hasselt et al. *Deep Reinforcement Learning with Double Q-learning.* AAAI, 2016. — Double DQN.
- Sutton & Barto. *Reinforcement Learning: An Introduction.* 2nd ed. — Q-learning, ε-greedy, fundamentos de RL.
