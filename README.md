# Learn2Slither — Implementación técnica del algoritmo de RL

Agente de Snake entrenado con **Deep Q-Learning (DQN)** sobre TensorFlow/Keras.
Este documento describe cómo funciona el algoritmo paso a paso y cómo lo
implementa el código actual.

El proyecto soporta **dos modos de entrenamiento**, seleccionables con la
opción `--training-mode` de la CLI:

- **`single`** (por defecto): un modelo dedicado a un único `--board-size`
  (10, 14 o 18). Codifica el estado con un **one-hot completo** del brazo
  visible — input grande pero atado a un tamaño concreto. Es la versión que
  mejor rinde en 10×10.
- **`multi`** (bonus de portabilidad): un solo modelo válido para los **tres
  tamaños 10/14/18**. Codifica el estado con **features de distancia
  tamaño-invariantes** (16 floats) y, durante el entrenamiento, cada episodio
  samplea un tamaño desde `[10, 14, 18]` con pesos `[0.7, 0.15, 0.15]`.

Las dos rutas comparten el mismo algoritmo (DQN + experience replay + target
network + Double DQN) y la misma función de recompensa. **Lo único que cambia
es la codificación del estado y la geometría de la red**. Cada vez que los dos
modos divergen lo señalamos explícitamente.

> Para uso e instalación, ver [USAGE.md](USAGE.md).

---

## 1. Arquitectura general

El código se organiza en cinco archivos:

| Archivo                                          | Responsabilidad                                                                              |
|--------------------------------------------------|----------------------------------------------------------------------------------------------|
| [src/AI_Model/main.py](src/AI_Model/main.py)     | CLI, bucle de entrenamiento/evaluación, sampling de tamaños en `multi`, checkpoints, carga de modelos. |
| [src/AI_Model/Agent.py](src/AI_Model/Agent.py)   | DQN: arquitectura de la red, replay buffer, target network, `process_view` con dispatch single/multi. |
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

con `γ = 0.98` ([Agent.py:56](src/AI_Model/Agent.py#L56)).

Las dos secciones siguientes explican los **dos pilares conceptuales** del
algoritmo: la **Q-function** (qué queremos calcular) y la **red neuronal**
(con qué la calculamos). En la sección 4 vemos cómo cambia la entrada según
el modo.

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

`γ = 0.98` es el discount factor: las recompensas lejanas valen menos que las
inmediatas. El agente prefiere una manzana ahora a una manzana en 100 pasos.

### 2.3 ¿Para qué sirve la Q-function?

**Para tomar decisiones.** Si conozco `Q(s, a)` para todas las acciones `a`,
mi mejor jugada es `argmax_a Q(s, a)`.

Esto es exactamente lo que hace
[Agent.get_action()](src/AI_Model/Agent.py#L221) cuando no está explorando:

```python
state = self.process_view(view, self.board_size)        # codifica el estado
q_values = self.model(state, training=False)            # forward pass
return int(np.argmax(q_values[0]))                      # elige la acción
                                                        # de mayor Q-value
```

**La política óptima sale gratis** una vez tienes una buena Q-function — y
eso vale por igual para `single` y `multi`: ambos modos producen un vector
de 4 Q-values en la salida.

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

En problemas pequeños, `Q` cabe en una **Q-table**. Aquí no:

- En modo `single` con visión cruz one-hot 240D (10×10) hay hasta `2²⁴⁰ ≈
  10⁷²` estados posibles.
- En modo `multi` el estado es un vector continuo de 16 floats — tampoco
  tabulable.

Aproximamos `Q(s, a)` con una **función paramétrica** `Q(s, a; θ)` que
generaliza entre estados parecidos. Esa función paramétrica es la red
neuronal.

---

## 3. La red neuronal — qué es, qué hace, para qué sirve aquí

### 3.1 ¿Qué hace exactamente la red en este proyecto?

Aproxima la Q-function. La forma del input cambia según el modo, pero la
salida es siempre la misma:

```
Modo single:  Entrada = vector one-hot de 4·N·6 floats
              (con N = board_size, 240 floats para 10×10)
Modo multi:   Entrada = vector de 16 floats (4 dirs × 4 features 1/d)

En ambos:     Salida   = vector de 4 floats
                       = [Q(s,UP), Q(s,DOWN), Q(s,LEFT), Q(s,RIGHT)]
```

**Una sola pasada hacia delante** (forward pass) nos da los Q-values de las
cuatro acciones a la vez.

### 3.2 Arquitectura — capa por capa

Definida en [Agent._build_model()](src/AI_Model/Agent.py#L93). El método
ramifica según `self.mode`:

#### Modo `single` — red grande para input rico

```
Input(4 · board_size · 6)        ← 240 para 10×10 (336 para 14, 432 para 18)
    │
    ▼
Dense(512, ReLU)
    │
Dropout(0.1)
    │
    ▼
Dense(256, ReLU)
    │
Dropout(0.1)
    │
    ▼
Dense(128, ReLU)
    │
    ▼
Dense(4, lineal)
                                  ─────────────────
                                  ≈ 288 132 parámetros (10×10)
```

#### Modo `multi` — red compacta para input ya comprimido

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

#### Por qué la diferencia de tamaño

| Aspecto                  | `single`                                | `multi`                                       |
|--------------------------|-----------------------------------------|-----------------------------------------------|
| Dimensión del input      | 240 / 336 / 432 según `board_size`       | 16 (fijo)                                     |
| Información por feature  | Indicador binario de categoría por celda | Distancia normalizada por entidad y dirección |
| Capacidad necesaria      | Alta — la red debe aprender a leer un one-hot disperso | Baja — las features ya son interpretables |
| Riesgo de overfitting    | Alto → dropout y red ancha               | Bajo → red estrecha sin dropout               |

El modo `multi` puede permitirse una red mucho más pequeña porque el input ya
contiene información destilada (¿a qué distancia hay pared? ¿a qué distancia
hay manzana?). En `single` la red tiene que aprender a leer el one-hot — una
tarea más cruda que requiere más capacidad.

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

#### Dropout 0.1 sólo en `single`

Apaga aleatoriamente el 10% de activaciones en las dos primeras capas durante
training, sólo en `single`. La red `multi` (~5 K parámetros) es demasiado
pequeña para necesitarlo: añadir dropout reduciría su capacidad por debajo de
lo útil. Sólo se aplica en training, no en evaluación
([get_action](src/AI_Model/Agent.py#L221) usa `training=False`).

#### Huber loss (δ = 1.0)

[Agent.py:74](src/AI_Model/Agent.py#L74): cuadrática para errores pequeños
(gradientes suaves cerca del óptimo) y lineal para errores grandes
(gradientes acotados). En DQN al inicio los Q-values están descalibrados;
con MSE, errores de magnitud 250 producirían gradientes proporcionales a
62500 → desestabilizan la red.

#### Adam con `lr = 0.0005` y `clipnorm = 1.0`

[Agent.py:67-69](src/AI_Model/Agent.py#L67):

- **Adam**: combina momentum y learning rate adaptativo por parámetro.
- **Learning rate 0.0005**: bajo a propósito; RL es propenso a olvido
  catastrófico.
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

[Board.get_snake_view()](src/AI_Model/Board.py#L121) devuelve cuatro listas
ordenadas de la celda más cercana a la más lejana. Cada celda contiene uno
de seis caracteres:

| Char | Significado     | Índice one-hot |
|------|------------------|----------------|
| `0`  | Celda vacía      | 0              |
| `W`  | Pared            | 1              |
| `S`  | Cuerpo serpiente | 2              |
| `G`  | Manzana verde    | 3              |
| `R`  | Manzana roja     | 4              |
| `H`  | Cabeza           | 5              |

La salida de `get_snake_view()` es **idéntica para los dos modos**. La
diferencia está en el siguiente paso:
[Agent.process_view()](src/AI_Model/Agent.py#L134) **convierte la visión en
un tensor para la red** y ramifica según `self.mode`:

```python
def process_view(self, view, board_size=None):
    if self.mode == 'multi':
        return self._process_view_features(view)
    return self._process_view_onehot(view, board_size or self.board_size)
```

### 4.1 Modo `single` — one-hot completo (240D para 10×10)

[Agent._process_view_onehot()](src/AI_Model/Agent.py#L149):

- Cada brazo se rellena hasta `board_size` celdas. Si el brazo no contiene
  pared, se inserta una **pared virtual** en la última posición — así la red
  siempre percibe un límite y el tamaño del input es fijo.
- Cada celda se codifica como un vector one-hot de 6 categorías
  (`CHAR_MAP` en [Agent.py:24](src/AI_Model/Agent.py#L24)).
- Resultado final: `4 brazos × board_size celdas × 6 categorías` valores.
- Para `board_size = 10` → **240 dimensiones** (336 para 14, 432 para 18).

> ¿Por qué one-hot y no enteros? Los enteros impondrían un orden ficticio
> (`G=3` "más cerca de" `R=4` que de `0`), sesgando el aprendizaje. One-hot
> trata cada categoría como independiente.

**Ventaja**: la red ve la posición exacta de cada elemento del brazo.
**Coste**: el input está ligado al `board_size`; cambiar el tamaño cambia la
arquitectura. **No portable.**

### 4.2 Modo `multi` — features tamaño-invariantes (16D)

[Agent._process_view_features()](src/AI_Model/Agent.py#L185) genera un vector
de **16 dimensiones que NO depende del tamaño del tablero**, condición
necesaria para el bonus de portabilidad.

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

#### Por qué `1/distancia` y no `distancia` cruda

1. **Acota el rango** a `[0, 1]` independientemente de `board_size`.
2. **Da más peso a lo cercano** — la diferencia entre "pared a 1" y "pared a
   2" (1.0 → 0.5) es enorme; entre "pared a 9" y "pared a 10" (0.111 →
   0.100) es despreciable. Refleja la urgencia táctica.
3. **`0.0` representa naturalmente "no visible"**.

#### Trade-off frente a `single`

| Ventaja                                         | Coste                                                    |
|-------------------------------------------------|----------------------------------------------------------|
| Mismo input shape para 10/14/18                 | Pierde información posicional fina                       |
| Una sola red sirve para todos los tamaños       | Menos capacidad → red más pequeña                        |
| Generaliza por composición de distancias        | Convergencia algo más lenta en 10×10 que `single`        |
| Robusto frente a tableros nuevos (interpolación) | No distingue varias manzanas en el mismo brazo (sólo la más cercana) |

### 4.3 Comparación lado a lado

| Aspecto                  | `single`                                       | `multi`                              |
|--------------------------|------------------------------------------------|--------------------------------------|
| Input shape              | `(1, 4·N·6)` con N = board_size                | `(1, 16)` siempre                    |
| Información              | Tipo de cada celda visible                     | Distancia a la primera de cada tipo  |
| Granularidad             | Por celda                                      | Por dirección                        |
| Tableros entrenables     | Uno fijo                                       | Cualquiera (10/14/18 vistos en el sampling) |
| Tableros jugables tras entrenar | Sólo el entrenado                       | Los tres                             |
| Codificador              | [Agent.py:149](src/AI_Model/Agent.py#L149)     | [Agent.py:185](src/AI_Model/Agent.py#L185) |

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
| Movimiento sin comer     | `-0.3`                | `NO_EAT`            | Urgencia por buscar comida         |
| Colisión / loop / suicidio (red apple en cuerpo vacío) | `-100` | `INSTANT_GAMEOVER` | Señal de muerte fuerte             |
| Timeout `max_steps`      | `-50`                 | (literal en main.py) | Episodio demasiado largo           |

La función de recompensa es **idéntica para los dos modos**. El snake percibe
los mismos estímulos sin importar cómo se codifique su visión.

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

[Agent.get_action()](src/AI_Model/Agent.py#L221):

```python
if np.random.rand() < epsilon:
    return np.random.randint(0, self.output_size)   # exploración
state = self.process_view(view, self.board_size)
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
   learning rate a `0.001`; los siguientes 170 vuelven a `ε = 0.01` y
   `lr = 0.0005`. El spike de lr permite que las transiciones nuevas
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
> (sampleados con peso 70/15/15 por episodio). Cada mini-batch tiene una
> distribución representativa, lo que **fuerza a la red a aprender una
> política que funcione en los tres tableros simultáneamente** en lugar de
> especializarse en uno.

> Importante: en el buffer se guardan **vistas crudas** (las listas de chars
> que devuelve `Board.get_snake_view()`), no los tensores codificados. La
> codificación se aplica al sacar las muestras del buffer (paso 5b en la
> sección 11). Esto reduce 4× la memoria del buffer en `single` y permite
> que `multi` mezcle transiciones de los tres tamaños sin problemas de
> compatibilidad.

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

Internamente `get_action` llama a `process_view`, que dispatcha en función de
`self.mode` (single → 240D, multi → 16D), hace un forward pass y devuelve el
`argmax`.

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
    states.append(self.process_view(sv, self.board_size)[0])
    next_states.append(self.process_view(nsv, self.board_size)[0])
    actions.append(a); rewards.append(r); dones.append(d)
```

→ Aplica el encoding (one-hot en `single`, features `1/d` en `multi`) a cada
visión y construye arrays NumPy. **Esta es la única línea del flujo donde el
modo afecta a las dimensiones del tensor.**

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
| 6b | Encoding del batch                | [Agent.py:267-279](src/AI_Model/Agent.py#L267)                 | **single u multi (`process_view`)**             |
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
│             ↑                                                   │
│             └─ AQUÍ se aplica single/multi: dispatch en Agent   │
│                                                                 │
│  Paso 3  ←  Recompensa r    (creada por el entorno)             │
│  Paso 4  ←  Detección de done (collision / loop / timeout)      │
│  Paso 5  ←  Replay buffer   (vistas crudas, no tensores)        │
│  Paso 6a ←  Replay buffer   (sampling aleatorio)                │
│  Paso 6b ←  process_view sobre el batch                         │
│             ↑                                                   │
│             └─ y AQUÍ — última divergencia entre los dos modos  │
│                                                                 │
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

Definida en [main.py:698-751](src/AI_Model/main.py#L698):

| Argumento          | Default | Descripción                                                       |
|--------------------|---------|-------------------------------------------------------------------|
| `--sessions`       | `100`   | Episodios de entrenamiento o partidas de evaluación.              |
| `--load`           | `None`  | Path a `.keras` para cargar (lee `*_metadata.json` si existe).    |
| `--save`           | `None`  | Path para el modelo final; si se omite se autogenera con timestamp. |
| `--dontlearn`      | `False` | Modo evaluación (no llama a `replay()`).                          |
| `--visual`         | `off`   | `on` activa la UI pygame (lobby + tablero).                       |
| `--step-by-step`   | `False` | Pausa cada paso (sólo evaluación).                                |
| `--board-size`     | `10`    | Choices `[10, 14, 18]`. En `multi`: sólo display fallback.         |
| `--training-mode`  | `single` | Choices `single` / `multi` (sección 13).                         |
| `--fps`            | `10`    | FPS de la visualización pygame.                                   |
| `--debug`          | `False` | Output detallado en los 2 primeros episodios.                     |

### 12.1 Entrenamiento desde cero

```bash
# single, 10×10
python src/AI_Model/main.py --sessions 10000 --board-size 10
# (--training-mode single es el default)

# multi (un modelo para 10/14/18)
python src/AI_Model/main.py --sessions 10000 --training-mode multi
```

### 12.2 Continuar entrenamiento desde un checkpoint

```bash
python src/AI_Model/main.py --sessions 5000 \
    --load models/10x10/best_snake_10x10_9000ep.keras
```

[main.py:761-812](src/AI_Model/main.py#L761) lee `*_metadata.json` para
recuperar `mode`, `board_size`, `total_episodes`. Si el modelo cargado tiene
`mode='single'` con un `board_size` distinto al CLI, **el modelo gana** y se
sobreescribe `--board-size`. Si el `mode` cargado no coincide con
`--training-mode`, gana también el del modelo (las arquitecturas son
incompatibles).

> Modelos antiguos sin `*_metadata.json` se asumen `mode='single'` con el
> `--board-size` actual de la CLI (compatibilidad hacia atrás).

### 12.3 Evaluación

```bash
# Sin UI (terminal)
python src/AI_Model/main.py --dontlearn --sessions 10 \
    --load models/multi/best_snake_multi_10000ep.keras

# Con UI (selector + tablero)
python src/AI_Model/main.py --dontlearn --visual on \
    --load models/multi/best_snake_multi_10000ep.keras

# Paso a paso (Enter / tecla entre pasos)
python src/AI_Model/main.py --dontlearn --visual on --step-by-step \
    --load models/10x10/best_snake_10x10_9000ep.keras
```

---

## 13. Bonus — portabilidad de tamaño (`--training-mode multi`)

El subject incluye como bonus que el mismo modelo entrenado pueda jugar en
tableros 10×10, 14×14 y 18×18. Esto se cubre con el modo `multi`. Las
secciones 3 y 4 explican las dos diferencias estructurales (input y red);
aquí van los detalles operativos.

### 13.1 Diferencias respecto al modo `single`

| Aspecto                  | `single` (default)                              | `multi` (bonus)                                |
|--------------------------|-------------------------------------------------|------------------------------------------------|
| Input de la red          | One-hot 240D (10×10) / 336D / 432D              | **16D fijo** (4 dirs × 4 features `1/d`)       |
| Arquitectura             | `512 → 256 → 128 → 4` con dropout (~288 K params) | `64 → 64 → 4` (~5 K params)                  |
| Tamaño de tablero entrenamiento | Fijo (`--board-size`)                    | Sampleado por episodio: 70% 10×10, 15% 14×14, 15% 18×18 |
| Carpeta de salida        | `models/<size>x<size>/`                          | `models/multi/`                                |
| `name_tag` en filename   | `<size>x<size>`                                  | `multi`                                        |
| Metadata `mode` en JSON  | `"single"`                                       | `"multi"`                                      |
| Metadata `board_size`    | El número (10/14/18)                              | `null`                                         |
| Botones en la UI         | 1 (sólo el tamaño entrenado)                    | 3 (selección libre)                            |

### 13.2 Sampling 70/15/15 — por qué no uniforme

[main.py:94-95](src/AI_Model/main.py#L94):

```python
MULTI_BOARD_SIZES = [10, 14, 18]
MULTI_BOARD_WEIGHTS = [0.7, 0.15, 0.15]
```

Si se entrenara uniformemente (33/33/33), la red dedicaría más gradiente del
necesario a 14×14 y 18×18, donde los episodios son **más largos y aportan
más transiciones por episodio**. El resultado es que el rendimiento en 10×10
— el "tablero canónico" del subject — cae respecto a la baseline `single`.

El reparto 70/15/15 mantiene la prioridad sobre 10×10 (caso de uso principal)
mientras expone a la red lo suficiente a 14/18 para que generalice.

### 13.3 Por qué la representación 16D es portable

La red `multi` aprende a partir de **distancias relativas**, que son
**invariantes a la longitud del brazo**. Un Q-value alto para "ir UP cuando
hay manzana verde a `1/3` de distancia y pared a `1/8`" tiene el mismo
significado en cualquier tamaño de tablero — la red no aprende posiciones,
sino **relaciones espaciales**.

El modo `single` no podía trasladarse: su input one-hot codifica
explícitamente la posición de cada celda del brazo, y la matriz de pesos de
la primera capa está dimensionada al `board_size` concreto. Cambiar de
10×10 a 14×14 implicaría literalmente otra red (otra forma de input → otro
número de pesos en la capa 1).

### 13.4 Flujo de selección de tamaño en runtime (evaluación)

```
                    ┌──────────────────────────┐
   --load model →   │   leer *_metadata.json   │
                    │   → mode, board_size     │
                    └────────────┬─────────────┘
                                 ▼
                    ┌──────────────────────────┐
                    │   Agent(mode=…)           │
                    │   con la red correcta    │
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

Todos los hiperparámetros del algoritmo (replay, target net, ε, recompensas)
son **idénticos para los dos modos**. La única diferencia está en la
arquitectura de la red.

| Parámetro                     | Valor               | Modo   | Ubicación                                                          |
|-------------------------------|---------------------|--------|--------------------------------------------------------------------|
| Learning rate base            | 0.0005              | ambos  | [Agent.py:67](src/AI_Model/Agent.py#L67)                           |
| Learning rate (spike cycles)  | 0.001               | ambos  | [main.py:420](src/AI_Model/main.py#L420)                           |
| Discount factor `γ`           | 0.98                | ambos  | [Agent.py:56](src/AI_Model/Agent.py#L56)                           |
| Replay buffer size            | 150 000             | ambos  | [Agent.py:81](src/AI_Model/Agent.py#L81)                           |
| Mini-batch                    | 512                 | ambos  | [Agent.py:84](src/AI_Model/Agent.py#L84)                           |
| Target update freq            | 1000                | ambos  | [Agent.py:85](src/AI_Model/Agent.py#L85)                           |
| ε inicial / final             | 1.0 / 0.01          | ambos  | [main.py:147,191](src/AI_Model/main.py#L147)                       |
| ε al continuar                | 0.005               | ambos  | [main.py:151](src/AI_Model/main.py#L151)                           |
| Warmup episodios              | 200                 | ambos  | [main.py:189](src/AI_Model/main.py#L189)                           |
| Decay end                     | 200 + min(35%·N, 4500) | ambos | [main.py:190](src/AI_Model/main.py#L190)                          |
| Cycle period / spike length   | 200 / 30            | ambos  | [main.py:417-418](src/AI_Model/main.py#L417)                       |
| Gradient clipping (norm)      | 1.0                 | ambos  | [Agent.py:67-69](src/AI_Model/Agent.py#L67)                        |
| Loss                          | Huber δ = 1         | ambos  | [Agent.py:74](src/AI_Model/Agent.py#L74)                           |
| Input size                    | `4·N·6` (240/336/432) | single | [Agent.py:62](src/AI_Model/Agent.py#L62)                          |
| Input size                    | `16` (fijo)         | multi  | [Agent.py:65](src/AI_Model/Agent.py#L65)                           |
| Capas ocultas                 | 512/256/128 + 2 dropout 0.1 | single | [Agent.py:101-110](src/AI_Model/Agent.py#L101)             |
| Capas ocultas                 | 64/64               | multi  | [Agent.py:111-117](src/AI_Model/Agent.py#L111)                     |
| Pesos por tamaño (sampling)   | 70/15/15 (10/14/18) | multi  | [main.py:95](src/AI_Model/main.py#L95)                             |
| `max_steps` (timeout)         | `N² × 20`           | ambos  | [main.py:218](src/AI_Model/main.py#L218)                           |
| `max_steps_without_food`      | `N² × max(2, len/6)` | ambos  | [main.py:269-273](src/AI_Model/main.py#L269)                       |

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

---

## 16. Referencias

- Mnih et al. *Human-level control through deep reinforcement learning.* Nature, 2015. — DQN base.
- Hasselt et al. *Deep Reinforcement Learning with Double Q-learning.* AAAI, 2016. — Double DQN.
- Sutton & Barto. *Reinforcement Learning: An Introduction.* 2nd ed. — Q-learning, ε-greedy, fundamentos de RL.
