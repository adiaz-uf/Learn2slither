# Fase 1: Fundamentos Teóricos - Reinforcement Learning

## 📚 Propósito de esta Fase

Antes de escribir cualquier línea de código, es fundamental comprender los conceptos teóricos que sustentan todo el proyecto. Esta fase establece la base conceptual necesaria para implementar correctamente un sistema de aprendizaje por refuerzo.

---

## 🎯 Conceptos Clave

### 1. Reinforcement Learning (RL) - Aprendizaje por Refuerzo

**¿Qué es?**
El Reinforcement Learning es un tipo de aprendizaje automático donde un agente aprende a tomar decisiones mediante la interacción con un entorno, recibiendo recompensas o penalizaciones por sus acciones.

**¿Para qué sirve en este proyecto?**
En Learn2Slither, la serpiente (agente) debe aprender a moverse por el tablero (entorno) para maximizar su longitud y evitar chocar. No tiene instrucciones explícitas sobre qué hacer; aprende por prueba y error.

**Papel en el proyecto:**
- Es el paradigma de aprendizaje que usarás para entrenar la serpiente
- Define cómo el agente interactúa con el entorno
- Establece el marco para el sistema de recompensas

---

### 2. Agente (Agent)

**¿Qué es?**
El agente es la entidad que toma decisiones y ejecuta acciones en el entorno. En este caso, es la serpiente que decide hacia dónde moverse.

**Características:**
- Tiene un estado actual (lo que "ve" del entorno)
- Puede ejecutar acciones (moverse en 4 direcciones)
- Recibe recompensas por sus acciones
- Aprende de la experiencia para mejorar sus decisiones

**Papel en el proyecto:**
- Representa la lógica de decisión de la serpiente
- Contiene la Q-Table o red neuronal que almacena el conocimiento aprendido
- Decide qué acción tomar basándose en el estado actual

**Relación con otros componentes:**
- Recibe información del **Entorno** (estado del tablero)
- Procesa esta información usando su **Función Q**
- Ejecuta una **Acción** que afecta al entorno
- Recibe una **Recompensa** que actualiza su conocimiento

---

### 3. Entorno (Environment)

**¿Qué es?**
El entorno es el mundo donde el agente existe y actúa. Incluye todas las reglas, estados posibles y consecuencias de las acciones.

**Componentes del entorno en Learn2Slither:**
- El tablero de juego (grid 10x10)
- La posición de la serpiente
- Las manzanas (verdes y rojas)
- Las reglas de colisión y game over
- El sistema de recompensas

**Papel en el proyecto:**
- Define el espacio de estados posibles
- Ejecuta las acciones del agente
- Calcula y entrega las recompensas
- Determina cuándo termina un episodio (game over)

**Responsabilidades:**
- Mantener el estado actual del juego
- Validar si las acciones son legales
- Actualizar el estado después de cada acción
- Detectar condiciones de fin de juego

---

### 4. El Ciclo de RL: Estado → Acción → Recompensa → Nuevo Estado

**¿Qué es?**
Este es el ciclo fundamental del aprendizaje por refuerzo. Describe cómo el agente interactúa continuamente con el entorno.

**Desglose del ciclo:**

1. **Estado ($S_t$):** La representación del entorno en el momento actual
   - En Learn2Slither: lo que la serpiente "ve" en las 4 direcciones
   - Ejemplo: "Hay una pared a la izquierda, comida verde arriba, cola propia abajo"

2. **Acción ($A_t$):** La decisión que toma el agente basándose en el estado
   - En Learn2Slither: moverse ARRIBA, ABAJO, IZQUIERDA o DERECHA
   - El agente elige la acción usando su función Q

3. **Recompensa ($R_t$):** El feedback que recibe el agente por su acción
   - Positiva: comer manzana verde (+puntos)
   - Negativa: chocar, comer manzana roja, perder tiempo (-puntos)

4. **Nuevo Estado ($S_{t+1}$):** El estado resultante después de la acción
   - El entorno se actualiza
   - La serpiente está en una nueva posición
   - El ciclo se repite

**Papel en el proyecto:**
- Define el flujo de datos entre componentes
- Establece cómo se comunica el Environment con el Agent
- Permite que el agente aprenda de cada interacción

**Ejemplo práctico:**
```
Estado: Serpiente ve comida verde a la derecha
  ↓
Acción: Agente decide moverse a la DERECHA
  ↓
Recompensa: +10 puntos (comió la comida)
  ↓
Nuevo Estado: Serpiente está en nueva posición, nueva comida aparece
  ↓
[El ciclo se repite]
```

---

### 5. Exploración vs Explotación

**¿Qué es?**
Es el dilema fundamental en RL: ¿debo explorar nuevas acciones o explotar lo que ya sé que funciona?

**Exploración:**
- Probar acciones nuevas o menos probadas
- Aprender sobre el entorno
- Necesario para descubrir mejores estrategias
- Ejemplo: Intentar moverse hacia una dirección nunca explorada

**Explotación:**
- Usar el conocimiento actual para maximizar recompensas
- Elegir acciones que ya sabemos que son buenas
- Necesario para obtener buenos resultados
- Ejemplo: Moverse siempre hacia donde sabemos que hay comida

**Balance en el proyecto:**
- Al inicio: Más exploración (el agente no sabe nada)
- Durante entrenamiento: Balance entre explorar y explotar
- Con modelo entrenado: Más explotación (usar lo aprendido)

**Papel en el proyecto:**
- Controla cómo el agente aprende durante el entrenamiento
- Se implementa mediante parámetros como epsilon (ε) en ε-greedy
- Permite que el agente descubra estrategias óptimas
- El modo `-dontlearn` usa solo explotación (sin aprender)

**¿Por qué es importante?**
- Sin exploración: El agente puede quedar atrapado en estrategias subóptimas
- Sin explotación: El agente nunca aprovecha lo aprendido
- Con balance correcto: El agente aprende estrategias óptimas

---

## 🧮 Algoritmo Q-Learning

### ¿Qué es Q-Learning?

Q-Learning es un algoritmo de aprendizaje por refuerzo que aprende la calidad (valor Q) de tomar una acción en un estado dado. Es un algoritmo "model-free" y "off-policy", lo que significa que no necesita conocer el modelo del entorno y puede aprender la política óptima.

### Función Q

**¿Qué es?**
La función Q, denotada como Q(s, a), representa el valor esperado de tomar la acción `a` en el estado `s` y luego seguir la política óptima.

**Interpretación:**
- Q(s, a) = "¿Qué tan bueno es hacer la acción `a` cuando estoy en el estado `s`?"
- Valores altos = acción prometedora
- Valores bajos = acción desfavorable

**Papel en el proyecto:**
- Es el "cerebro" del agente
- Almacena todo el conocimiento aprendido
- Se usa para decidir qué acción tomar

### Q-Table vs Red Neuronal

**Q-Table (Tabla Q):**
- Estructura de datos simple: tabla bidimensional
- Filas = estados posibles
- Columnas = acciones posibles
- Valores = Q(s, a) para cada par estado-acción
- Ventaja: Simple, fácil de entender y depurar
- Desventaja: No escala bien con muchos estados

**Red Neuronal (Deep Q-Network - DQN):**
- Aproxima la función Q usando una red neuronal
- Entrada: estado
- Salida: valores Q para cada acción
- Ventaja: Puede manejar espacios de estados muy grandes
- Desventaja: Más complejo, requiere más recursos

**Para este proyecto:**
- Puedes usar Q-Table si el espacio de estados es manejable
- Puedes usar DQN si quieres más sofisticación
- **Restricción:** Solo Q-Learning o Deep Q-Learning están permitidos

### Actualización de la Q-Table

**¿Cómo funciona?**
Después de cada acción, la Q-Table se actualiza usando la ecuación de Bellman:

```
Q(s, a) = Q(s, a) + α [R + γ * max(Q(s', a')) - Q(s, a)]
```

**Componentes:**
- **α (alpha) - Tasa de aprendizaje:** Qué tan rápido aprende (0 < α ≤ 1)
  - Alto: Aprende rápido pero puede ser inestable
  - Bajo: Aprende lento pero más estable
  
- **R - Recompensa inmediata:** La recompensa recibida por la acción
  
- **γ (gamma) - Factor de descuento:** Qué tan importante es el futuro (0 ≤ γ ≤ 1)
  - Alto: Valora recompensas futuras
  - Bajo: Solo valora recompensas inmediatas
  
- **max(Q(s', a')) - Mejor acción futura:** El mejor valor Q en el nuevo estado

**Papel en el proyecto:**
- Esta actualización es el "aprendizaje" del agente
- Se ejecuta después de cada acción
- Mejora gradualmente las decisiones del agente
- Con el tiempo, la Q-Table converge a valores óptimos

**Ejemplo conceptual:**
```
Estado actual: Serpiente ve comida a la derecha
Acción elegida: Moverse a la DERECHA
Recompensa: +10 (comió la comida)
Nuevo estado: Serpiente en nueva posición

Actualización:
- Si Q(estado_anterior, DERECHA) era 5
- Y la recompensa fue 10
- Y el mejor futuro es 8
- Entonces: Q(estado_anterior, DERECHA) aumenta hacia 10 + 8 = 18
```

---

## 🔗 Conexión entre Conceptos

**Flujo completo del aprendizaje:**

1. **Entorno** genera un **Estado** (lo que la serpiente ve)
2. **Agente** consulta su **Función Q** para evaluar acciones
3. **Agente** elige una **Acción** (balanceando exploración/explotación)
4. **Entorno** ejecuta la acción y calcula la **Recompensa**
5. **Entorno** genera el **Nuevo Estado**
6. **Agente** actualiza su **Función Q** usando la recompensa
7. El ciclo se repite

**Resultado:**
Con el tiempo, la Función Q aprende qué acciones son mejores en cada estado, permitiendo que la serpiente juegue de manera inteligente.

---

## 📝 Resumen de Conceptos Clave

| Concepto | ¿Qué es? | Papel en el Proyecto |
|----------|----------|---------------------|
| **Agente** | La serpiente que toma decisiones | Contiene la lógica de decisión y la Q-Table |
| **Entorno** | El tablero y reglas del juego | Gestiona el estado del juego y las recompensas |
| **Estado** | Lo que la serpiente "ve" | Entrada para la función Q |
| **Acción** | Movimiento (4 direcciones) | Salida de la función Q |
| **Recompensa** | Feedback positivo/negativo | Guía el aprendizaje |
| **Función Q** | Tabla/Red que almacena conocimiento | Cerebro del agente |
| **Exploración** | Probar cosas nuevas | Descubrir estrategias |
| **Explotación** | Usar conocimiento actual | Maximizar recompensas |

---

## 🎓 Siguiente Paso

Una vez comprendidos estos conceptos teóricos, estarás listo para la **Fase 2: Construcción del Entorno**, donde implementarás el tablero de juego que servirá como el entorno para tu agente.
