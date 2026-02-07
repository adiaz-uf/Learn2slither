# Fase 3: El Agente y la Visión (El Cerebro)

## 📚 Propósito de esta Fase

Esta es la fase crítica donde conectas la lógica de Reinforcement Learning con el juego. Aquí defines cómo el agente percibe el mundo (visión limitada), qué acciones puede tomar, y cómo se recompensan sus decisiones. Esta fase transforma el entorno físico en un problema de aprendizaje que el agente puede resolver.

---

## 🎯 Conceptos Fundamentales

### ¿Qué es la Visión del Agente?

**Definición:**
La visión es la representación del estado del entorno desde la perspectiva del agente. En Learn2Slither, la serpiente tiene una visión **limitada** - solo puede ver en las 4 direcciones cardinales desde su cabeza.

**Restricción Crítica:**
- ❌ **NO** puede ver todo el tablero
- ❌ **NO** puede tener información global
- ✅ **SÍ** solo puede ver en 4 direcciones: ARRIBA, ABAJO, IZQUIERDA, DERECHA
- ✅ **SÍ** debe usar solo lo que "ve" directamente desde su cabeza

**¿Por qué esta restricción?**
- Hace el problema más realista (como visión limitada en la naturaleza)
- Aumenta la dificultad del aprendizaje
- Evita soluciones triviales
- **Penalización severa** si se viola esta restricción

**Papel en el proyecto:**
- Define qué información recibe el agente
- Limita el espacio de estados posibles
- Determina la complejidad del problema de aprendizaje
- Es la entrada para la función Q del agente

---

## 👁️ Definir el Estado (Visión)

### 1. Sistema de Visión en 4 Direcciones

**¿Qué ve la serpiente?**
Desde la posición de su cabeza, la serpiente "mira" en cada una de las 4 direcciones hasta encontrar algo o llegar al borde.

**Direcciones:**
- **ARRIBA (UP):** Hacia arriba desde la cabeza
- **ABAJO (DOWN):** Hacia abajo desde la cabeza
- **IZQUIERDA (LEFT):** Hacia la izquierda desde la cabeza
- **DERECHA (RIGHT):** Hacia la derecha desde la cabeza

**¿Qué puede detectar en cada dirección?**
En cada dirección, la serpiente puede encontrar:
- **W (Wall):** Una pared (borde del tablero)
- **H (Head):** Su propia cabeza (no debería pasar normalmente)
- **S (Snake/Body):** Parte de su propio cuerpo/cola
- **G (Green):** Una manzana verde
- **R (Red):** Una manzana roja
- **0 (Empty):** Nada (celda vacía)

**Mecánica de detección:**
1. Desde la cabeza, trazar una línea en la dirección
2. La primera cosa encontrada es lo que se reporta
3. Si no se encuentra nada hasta el borde, se reporta W (pared)

**Ejemplo conceptual:**
```
Tablero:
[ ][ ][G][ ]
[ ][S][S][ ]
[ ][S][H][R]  <- H = Cabeza
[ ][ ][ ][ ]

Visión desde H:
- ARRIBA: Encuentra S (cuerpo propio) a distancia 1
- ABAJO: Encuentra 0 (vacío) a distancia 1, luego W (pared) a distancia 2
- IZQUIERDA: Encuentra S (cuerpo propio) a distancia 1
- DERECHA: Encuentra R (manzana roja) a distancia 1
```

---

### 2. Formato de Salida (Output de Terminal)

**¿Qué es?**
La representación textual de lo que la serpiente ve, usando caracteres específicos.

**Formato requerido:**
Debe seguir el formato mostrado en los ejemplos del sujeto, usando los caracteres:
- **W** = Wall (Pared)
- **H** = Head (Cabeza propia)
- **S** = Snake/Body (Cuerpo propio)
- **G** = Green apple (Manzana verde)
- **R** = Red apple (Manzana roja)
- **0** = Empty (Vacío)

**Estructura del output:**
Puede ser una cadena o estructura que represente las 4 direcciones:
```
UP: G    (manzana verde arriba)
DOWN: 0  (vacío abajo)
LEFT: S  (cuerpo propio a la izquierda)
RIGHT: W (pared a la derecha)
```

O en formato más compacto:
```
[G, 0, S, W]  # [UP, DOWN, LEFT, RIGHT]
```

**Papel en el proyecto:**
- Es la representación del estado que el agente usa
- Debe ser consistente y parseable
- Facilita el debugging (puedes ver qué "ve" el agente)
- Es la entrada para la función Q

**Consideraciones:**
- Debe ser fácil de convertir a un formato numérico para la Q-Table
- Debe ser único para cada situación visual distinta
- Debe ser eficiente de calcular

---

### 3. Distancia (Opcional pero Útil)

**¿Qué es?**
Además de saber QUÉ hay en cada dirección, puede ser útil saber A QUÉ DISTANCIA está.

**Información de distancia:**
- Distancia en celdas hasta el objeto detectado
- Ejemplo: "Manzana verde a 3 celdas de distancia arriba"

**¿Es necesario?**
- No está explícitamente requerido en las especificaciones
- Pero puede mejorar significativamente el aprendizaje
- Permite al agente priorizar objetivos cercanos

**Representación:**
```
UP: (G, 3)    # Manzana verde a 3 celdas arriba
DOWN: (0, 1)  # Vacío, pared a 1 celda abajo
LEFT: (S, 1)  # Cuerpo propio a 1 celda izquierda
RIGHT: (W, 0) # Pared inmediatamente a la derecha
```

**Papel en el proyecto:**
- Enriquece la información del estado
- Permite decisiones más sofisticadas
- Aumenta el espacio de estados (más combinaciones posibles)

---

## 🎮 Definir las Acciones

### Espacio de Acciones

**¿Qué es?**
El conjunto de todas las acciones posibles que el agente puede tomar.

**En Learn2Slither:**
El agente tiene exactamente **4 acciones posibles:**
1. **ARRIBA (UP):** Mover la cabeza hacia arriba
2. **ABAJO (DOWN):** Mover la cabeza hacia abajo
3. **IZQUIERDA (LEFT):** Mover la cabeza hacia la izquierda
4. **DERECHA (RIGHT):** Mover la cabeza hacia la derecha

**Restricciones:**
- No puede moverse en diagonal
- No puede "no moverse" (debe elegir una dirección)
- No puede moverse fuera de los límites (causa game over)

**Representación:**
Puede ser:
- Números: `0=UP, 1=DOWN, 2=LEFT, 3=RIGHT`
- Strings: `"UP", "DOWN", "LEFT", "RIGHT"`
- Vectores: `[1,0,0,0]` para UP, `[0,1,0,0]` para DOWN, etc.

**Papel en el proyecto:**
- Define el espacio de salida de la función Q
- La función Q debe evaluar estas 4 acciones para cada estado
- El agente elige la acción con mayor valor Q

**Conexión con la Q-Table:**
- Si usas Q-Table: Columnas = 4 acciones
- Si usas DQN: Salida de la red = 4 valores (uno por acción)

---

## 🎁 Sistema de Recompensas (Rewards)

### ¿Qué es un Sistema de Recompensas?

**Definición:**
Un sistema que asigna valores numéricos (recompensas) a las acciones del agente, indicando qué tan "buenas" o "malas" son.

**Propósito:**
- Guiar el aprendizaje del agente
- Enseñar qué comportamientos son deseables
- Penalizar comportamientos indeseables
- Permitir que el agente descubra estrategias óptimas

**Papel en el proyecto:**
- Es el "feedback" que recibe el agente
- Se usa para actualizar la función Q
- Determina qué estrategias aprende el agente
- Debe estar bien diseñado para que el agente aprenda correctamente

---

### Tipos de Recompensas

#### 1. Recompensa Positiva

**¿Cuándo se otorga?**
- **Comer manzana verde:** Cuando la serpiente come una manzana verde

**Valor típico:**
- Puede ser `+10`, `+20`, `+100`, etc.
- Debe ser suficientemente alta para motivar al agente
- Debe ser mayor que las recompensas negativas

**Efecto en el aprendizaje:**
- Aumenta el valor Q de la acción que llevó a comer la manzana
- Hace que el agente prefiera estados donde hay manzanas verdes cerca
- Guía al agente hacia comportamientos beneficiosos

**Consideraciones de diseño:**
- No debe ser tan alta que ignore otros factores
- Debe balancearse con otras recompensas
- Puede variar según la distancia a la manzana (más cerca = más recompensa)

---

#### 2. Recompensas Negativas

**¿Cuándo se otorgan?**
Hay tres situaciones que generan recompensas negativas:

**a) Comer manzana roja:**
- Cuando la serpiente come una manzana roja
- Valor típico: `-10`, `-20`, `-50`
- Efecto: Desalienta comer manzanas rojas

**b) Chocar/Morir:**
- Cuando la serpiente choca con pared o su cuerpo
- Cuando ocurre game over
- Valor típico: `-100`, `-200`, `-1000` (muy negativo)
- Efecto: Fuertemente desalienta acciones que causan muerte

**c) Perder el tiempo (comer nada):**
- Cuando la serpiente se mueve sin comer nada
- Valor típico: `-1`, `-0.1`, `-0.5` (ligeramente negativo)
- Efecto: Motiva al agente a ser eficiente y buscar comida

**Efecto en el aprendizaje:**
- Disminuye el valor Q de acciones que llevan a estos resultados
- Enseña al agente a evitar comportamientos peligrosos
- Guía al agente hacia estrategias seguras y eficientes

**Consideraciones de diseño:**
- Las recompensas negativas deben ser proporcionales a la severidad
- Morir debe ser mucho más negativo que perder tiempo
- Deben balancearse con las recompensas positivas

---

### Diseño del Sistema de Recompensas

**Principios importantes:**

1. **Escala apropiada:**
   - Las recompensas deben estar en una escala razonable
   - No demasiado grandes (pueden causar inestabilidad)
   - No demasiado pequeñas (pueden ser ignoradas)

2. **Balance:**
   - Recompensas positivas vs negativas
   - Recompensas inmediatas vs futuras
   - Diferentes tipos de recompensas negativas

3. **Claridad:**
   - Cada acción debe tener una recompensa clara
   - El agente debe poder asociar acciones con resultados
   - Evitar recompensas ambiguas

4. **Motivación:**
   - Debe motivar al agente a lograr el objetivo (crecer)
   - Debe desalentar comportamientos peligrosos
   - Debe promover eficiencia

**Ejemplo de sistema de recompensas:**
```
Comer manzana verde:    +10
Comer manzana roja:     -10
Chocar/Morir:          -100
Moverse sin comer:      -0.1
```

**Ajuste fino:**
- Puede requerir experimentación
- Diferentes valores pueden afectar el aprendizaje
- Debe probarse con diferentes configuraciones

---

## 🔄 Flujo de Información: Visión → Decisión → Acción

### Proceso Completo

1. **El Entorno proporciona el estado completo:**
   - Posición de la serpiente
   - Posición de todas las manzanas
   - Estado del tablero completo

2. **El Interpreter traduce a visión limitada:**
   - Toma el estado completo
   - Calcula qué ve la serpiente en 4 direcciones
   - Genera el formato de salida (W, H, S, G, R, 0)

3. **El Agente recibe la visión:**
   - Esta es la entrada para la función Q
   - El agente no conoce el estado completo, solo la visión

4. **El Agente consulta su función Q:**
   - Para cada acción posible, consulta Q(visión, acción)
   - Obtiene 4 valores Q (uno por acción)

5. **El Agente elige una acción:**
   - Puede elegir la acción con mayor Q (explotación)
   - O una acción aleatoria (exploración)
   - O balancear ambos (ε-greedy)

6. **El Entorno ejecuta la acción:**
   - Mueve la serpiente
   - Calcula la recompensa
   - Genera el nuevo estado

7. **El Agente aprende:**
   - Actualiza su función Q usando la recompensa
   - Mejora su conocimiento para futuras decisiones

---

## 🧠 El Módulo Interpreter

### ¿Qué es el Interpreter?

**Definición:**
El módulo que traduce el estado completo del entorno a la visión limitada del agente.

**Responsabilidades:**
1. Recibir el estado completo del entorno
2. Calcular qué ve la serpiente en cada dirección
3. Formatear la visión según el formato requerido
4. Proporcionar la visión al agente

**Input:**
- Estado completo del tablero
- Posición de la cabeza de la serpiente
- Posiciones de todas las manzanas
- Posiciones de todos los segmentos de la serpiente

**Output:**
- Visión en 4 direcciones
- Formato: [UP, DOWN, LEFT, RIGHT] con valores (W, H, S, G, R, 0)
- Opcionalmente: distancias

**Algoritmo conceptual:**
```
Para cada dirección (UP, DOWN, LEFT, RIGHT):
  1. Partir desde la posición de la cabeza
  2. Mover en esa dirección celda por celda
  3. En cada celda, verificar qué hay:
     - Si es borde del tablero → W (Wall)
     - Si es parte del cuerpo → S (Snake)
     - Si es manzana verde → G (Green)
     - Si es manzana roja → R (Red)
     - Si es vacío → continuar
  4. Reportar el primer objeto encontrado
  5. Si no se encuentra nada hasta el borde → W (Wall)
```

**Papel en el proyecto:**
- Es el "traductor" entre el entorno y el agente
- Implementa la restricción de visión limitada
- Es crítico para el aprendizaje correcto
- Debe ser eficiente (se llama en cada paso)

---

## 🔗 Conexión con Otras Fases

### Con Fase 1 (Fundamentos):
- La visión es el **Estado ($S_t$)** del ciclo RL
- Las acciones son las **Acciones ($A_t$)** del ciclo RL
- Las recompensas son las **Recompensas ($R_t$)** del ciclo RL

### Con Fase 2 (Entorno):
- El entorno proporciona el estado completo
- El interpreter traduce a visión limitada
- El entorno ejecuta las acciones del agente

### Con Fase 4 (Estructura Modular):
- El Interpreter es un módulo separado
- Se comunica entre Environment y Agent
- Sigue el flujo: Environment → Interpreter → Agent

---

## 📝 Resumen de Conceptos Clave

| Concepto | Descripción | Papel en el Proyecto |
|----------|-------------|---------------------|
| **Visión Limitada** | Solo 4 direcciones desde la cabeza | Define el espacio de estados |
| **Formato de Salida** | W, H, S, G, R, 0 | Representación del estado |
| **4 Acciones** | UP, DOWN, LEFT, RIGHT | Espacio de acciones |
| **Recompensa Positiva** | Comer manzana verde | Guía hacia objetivos |
| **Recompensas Negativas** | Roja, chocar, perder tiempo | Desalienta malos comportamientos |
| **Interpreter** | Traduce estado completo a visión | Implementa restricción de visión |

---

## ⚠️ Advertencias Importantes

1. **NO proporcionar información global:**
   - El agente NO debe conocer posiciones absolutas
   - El agente NO debe ver todo el tablero
   - **Penalización severa** si se viola esto

2. **Formato de salida correcto:**
   - Debe seguir el formato especificado
   - Debe usar los caracteres correctos (W, H, S, G, R, 0)
   - Debe ser consistente

3. **Sistema de recompensas balanceado:**
   - Debe motivar comportamientos correctos
   - Debe desalentar comportamientos peligrosos
   - Puede requerir ajuste fino

---

## 🎓 Siguiente Paso

Una vez definida la visión, las acciones y las recompensas, estarás listo para la **Fase 4: Estructura Técnica Modular**, donde organizarás todo el código en módulos bien definidos que se comunican entre sí.
