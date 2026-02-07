# Fase 4: Estructura Técnica Modular

## 📚 Propósito de esta Fase

Esta fase se enfoca en organizar el código en una arquitectura modular bien definida. La modularidad facilita el desarrollo, el mantenimiento, la evaluación y permite que los componentes trabajen juntos de manera clara y eficiente.

---

## 🎯 Conceptos Fundamentales

### ¿Qué es una Arquitectura Modular?

**Definición:**
Una arquitectura modular es una forma de organizar el código dividiéndolo en módulos independientes, cada uno con responsabilidades específicas y bien definidas.

**Características:**
- **Separación de responsabilidades:** Cada módulo tiene un propósito claro
- **Interfaz bien definida:** Cada módulo sabe cómo comunicarse con otros
- **Independencia:** Los módulos pueden desarrollarse y probarse por separado
- **Reutilización:** Los módulos pueden usarse en diferentes contextos

**Ventajas:**
- Facilita el desarrollo (puedes trabajar en un módulo a la vez)
- Facilita el debugging (problemas aislados en módulos específicos)
- Facilita la evaluación (cada módulo puede probarse independientemente)
- Facilita el mantenimiento (cambios en un módulo no afectan otros)
- Facilita la comprensión (código organizado y claro)

**Papel en el proyecto:**
- Organiza todo el código de manera lógica
- Permite que diferentes personas trabajen en diferentes partes
- Facilita la evaluación del proyecto
- Hace el código más profesional y mantenible

---

## 🏗️ Módulos Requeridos

### 1. Módulo Environment (Entorno)

**¿Qué es?**
El módulo que gestiona el tablero de juego y todas las reglas del juego.

**Responsabilidades:**
1. **Gestionar el estado del tablero:**
   - Mantener la posición de la serpiente
   - Mantener las posiciones de las manzanas
   - Mantener el estado de cada celda del tablero

2. **Ejecutar acciones:**
   - Recibir acciones del agente
   - Validar que las acciones son legales
   - Mover la serpiente según la acción
   - Actualizar el estado del tablero

3. **Calcular recompensas:**
   - Determinar qué recompensa corresponde a cada acción
   - Considerar si se comió comida, si hubo colisión, etc.

4. **Detectar condiciones de fin:**
   - Verificar colisiones con paredes
   - Verificar colisiones con el cuerpo
   - Verificar si la longitud llegó a 0

5. **Gestionar elementos del juego:**
   - Reposicionar comida cuando es comida
   - Asegurar que siempre hay 2 manzanas verdes y 1 roja
   - Inicializar el tablero para nuevos episodios

**Interfaz (Métodos principales):**
- `reset()`: Reinicia el entorno para un nuevo episodio
- `step(action)`: Ejecuta una acción y retorna (nuevo_estado, recompensa, terminado, info)
- `get_state()`: Retorna el estado completo del tablero
- `render()`: Actualiza la visualización (si está activada)

**Input:**
- Acciones del agente (UP, DOWN, LEFT, RIGHT)

**Output:**
- Estado completo del tablero
- Recompensas
- Información sobre si el juego terminó
- Información adicional (puntuación, longitud, etc.)

**Papel en el proyecto:**
- Es el "mundo" donde vive el agente
- Implementa todas las reglas del juego
- Proporciona la información necesaria para el aprendizaje
- No contiene lógica de aprendizaje (solo reglas del juego)

**Relación con otros módulos:**
- Recibe acciones del **Módulo Agent**
- Proporciona estado completo al **Módulo Interpreter**
- No conoce nada sobre la visión limitada o la Q-Table

---

### 2. Módulo Interpreter (Intérprete)

**¿Qué es?**
El módulo que traduce el estado completo del entorno a la visión limitada del agente.

**Responsabilidades:**
1. **Recibir el estado completo:**
   - Obtener el estado completo del módulo Environment
   - Acceder a la posición de la cabeza de la serpiente
   - Acceder a las posiciones de todos los elementos

2. **Calcular la visión:**
   - Para cada dirección (UP, DOWN, LEFT, RIGHT):
     - Trazar una línea desde la cabeza
     - Detectar el primer objeto encontrado
     - Determinar la distancia (opcional)

3. **Formatear la visión:**
   - Convertir la información detectada al formato requerido
   - Usar los caracteres correctos (W, H, S, G, R, 0)
   - Estructurar la información de manera consistente

4. **Proporcionar la visión al agente:**
   - Retornar la visión en el formato que el agente espera
   - Asegurar que solo se proporciona información limitada

**Interfaz (Métodos principales):**
- `interpret(state)`: Toma el estado completo y retorna la visión limitada
- `get_vision_from_position(state, position)`: Calcula la visión desde una posición específica
- `format_vision(vision_data)`: Formatea la visión según el formato requerido

**Input:**
- Estado completo del tablero (del módulo Environment)
- Posición de la cabeza de la serpiente

**Output:**
- Visión limitada en 4 direcciones
- Formato: [UP, DOWN, LEFT, RIGHT] con valores (W, H, S, G, R, 0)
- Opcionalmente: distancias a cada objeto

**Papel en el proyecto:**
- Implementa la restricción de visión limitada
- Es el "traductor" entre el entorno y el agente
- Asegura que el agente solo recibe información permitida
- Es crítico para el aprendizaje correcto

**Relación con otros módulos:**
- Recibe estado completo del **Módulo Environment**
- Proporciona visión limitada al **Módulo Agent**
- No conoce nada sobre la Q-Table o el aprendizaje

**Restricciones importantes:**
- **NO** debe proporcionar información global
- **NO** debe dar posiciones absolutas
- **SÍ** debe limitarse a las 4 direcciones desde la cabeza
- **SÍ** debe seguir el formato especificado

---

### 3. Módulo Agent (Agente)

**¿Qué es?**
El módulo que contiene la lógica de aprendizaje y toma de decisiones del agente.

**Responsabilidades:**
1. **Gestionar la función Q:**
   - Almacenar la Q-Table o la red neuronal
   - Inicializar los valores Q
   - Actualizar los valores Q después de cada acción

2. **Tomar decisiones:**
   - Recibir la visión del módulo Interpreter
   - Consultar la función Q para cada acción posible
   - Elegir una acción (balanceando exploración/explotación)

3. **Aprender:**
   - Recibir recompensas del módulo Environment
   - Actualizar la función Q usando la ecuación de Bellman
   - Mejorar las decisiones futuras basándose en la experiencia

4. **Gestionar el aprendizaje:**
   - Controlar la exploración vs explotación (ε-greedy)
   - Gestionar la tasa de aprendizaje (α)
   - Gestionar el factor de descuento (γ)

5. **Guardar y cargar modelos:**
   - Exportar la Q-Table/red neuronal a archivo
   - Cargar modelos previamente entrenados
   - Continuar el entrenamiento desde un modelo guardado

**Interfaz (Métodos principales):**
- `get_action(state)`: Toma la visión y retorna una acción
- `learn(state, action, reward, next_state, done)`: Actualiza la función Q
- `save_model(filename)`: Guarda el modelo a un archivo
- `load_model(filename)`: Carga un modelo desde un archivo
- `set_learning_mode(enabled)`: Activa/desactiva el aprendizaje

**Input:**
- Visión limitada (del módulo Interpreter)
- Recompensas (del módulo Environment)
- Información sobre si el episodio terminó

**Output:**
- Acciones (UP, DOWN, LEFT, RIGHT)
- Modelos guardados (archivos .txt)

**Papel en el proyecto:**
- Es el "cerebro" del agente
- Contiene todo el conocimiento aprendido
- Toma todas las decisiones
- Es donde ocurre el aprendizaje

**Relación con otros módulos:**
- Recibe visión del **Módulo Interpreter**
- Envía acciones al **Módulo Environment**
- No conoce el estado completo del tablero (solo la visión)

**Componentes internos:**
- **Q-Table o Red Neuronal:** Almacena el conocimiento
- **Algoritmo de actualización:** Ecuación de Bellman
- **Estrategia de exploración:** ε-greedy u otra
- **Parámetros de aprendizaje:** α (tasa), γ (descuento), ε (exploración)

---

## 🔄 Ciclo de Comunicación

### Flujo Completo de Datos

El flujo de comunicación entre módulos sigue este ciclo:

```
Environment → State → Interpreter → Agent → Action → Environment
```

**Desglose paso a paso:**

1. **Environment genera un Estado:**
   - El entorno tiene el estado completo del tablero
   - Incluye posiciones de serpiente, manzanas, etc.

2. **Interpreter traduce Estado a Visión:**
   - Recibe el estado completo del Environment
   - Calcula qué ve la serpiente en 4 direcciones
   - Retorna la visión limitada

3. **Agent recibe Visión y decide Acción:**
   - Recibe la visión del Interpreter
   - Consulta su función Q
   - Elige una acción (UP, DOWN, LEFT, RIGHT)

4. **Environment ejecuta Acción:**
   - Recibe la acción del Agent
   - Mueve la serpiente
   - Calcula la recompensa
   - Genera el nuevo estado
   - Retorna (nuevo_estado, recompensa, terminado, info)

5. **Agent aprende:**
   - Recibe la recompensa y el nuevo estado
   - Actualiza su función Q
   - Mejora para futuras decisiones

6. **El ciclo se repite:**
   - Nuevo estado → Nueva visión → Nueva acción → Nueva recompensa

**Visualización del flujo:**
```
┌─────────────┐
│ Environment │  (Estado completo del tablero)
└──────┬──────┘
       │ get_state()
       ↓
┌─────────────┐
│ Interpreter │  (Traduce a visión limitada)
└──────┬──────┘
       │ interpret()
       ↓
┌─────────────┐
│   Agent     │  (Consulta Q-Table, elige acción)
└──────┬──────┘
       │ get_action()
       ↓
┌─────────────┐
│ Environment │  (Ejecuta acción, calcula recompensa)
└──────┬──────┘
       │ step(action)
       ↓
┌─────────────┐
│   Agent     │  (Aprende, actualiza Q-Table)
└─────────────┘
```

---

## 🔌 Interfaces entre Módulos

### Environment ↔ Interpreter

**Environment proporciona:**
- Estado completo del tablero
- Posición de la cabeza de la serpiente
- Posiciones de todas las manzanas
- Posiciones de todos los segmentos de la serpiente

**Interpreter necesita:**
- Acceso al estado completo
- Método para obtener información de celdas específicas
- Información sobre los límites del tablero

**Formato de comunicación:**
- El Environment puede tener un método `get_state()` que retorna un objeto/diccionario con toda la información
- El Interpreter llama a este método y procesa la información

---

### Interpreter ↔ Agent

**Interpreter proporciona:**
- Visión limitada en 4 direcciones
- Formato: [UP, DOWN, LEFT, RIGHT] con valores (W, H, S, G, R, 0)
- Opcionalmente: distancias

**Agent necesita:**
- Visión en un formato que pueda usar como clave para la Q-Table
- Información consistente y parseable

**Formato de comunicación:**
- El Interpreter tiene un método `interpret(state)` que retorna la visión
- El Agent llama a este método y usa el resultado para consultar la Q-Table

---

### Agent ↔ Environment

**Agent proporciona:**
- Acciones (UP, DOWN, LEFT, RIGHT)
- Puede ser un número (0-3) o string ("UP", "DOWN", etc.)

**Environment necesita:**
- Acción en un formato que pueda ejecutar
- Validación de que la acción es legal

**Formato de comunicación:**
- El Agent tiene un método `get_action(state)` que retorna una acción
- El Environment tiene un método `step(action)` que ejecuta la acción

**Environment proporciona (retroalimentación):**
- Recompensa
- Nuevo estado
- Información sobre si terminó
- Información adicional (puntuación, longitud, etc.)

**Agent necesita:**
- Recompensa para actualizar la Q-Table
- Nuevo estado para calcular el valor futuro
- Información sobre terminación para manejar episodios

**Formato de comunicación:**
- El Environment retorna `(new_state, reward, done, info)` de `step(action)`
- El Agent usa esta información en `learn(state, action, reward, next_state, done)`

---

## 📁 Organización del Código

### Estructura de Archivos Recomendada

```
src/
├── Environment/
│   ├── __init__.py
│   └── game_board.py      # Módulo Environment
├── Interpreter/
│   ├── __init__.py
│   └── vision.py          # Módulo Interpreter
├── Agent/
│   ├── __init__.py
│   ├── q_learning.py      # Módulo Agent (Q-Table)
│   └── model_io.py        # Guardar/cargar modelos
└── main.py                # Orquesta todos los módulos
```

**Ventajas de esta estructura:**
- Cada módulo está en su propia carpeta
- Fácil de encontrar y modificar cada componente
- Fácil de probar cada módulo independientemente
- Fácil de evaluar

---

### Principios de Diseño Modular

1. **Alta cohesión:**
   - Cada módulo debe tener responsabilidades relacionadas
   - Todas las funciones de un módulo deben trabajar juntas

2. **Bajo acoplamiento:**
   - Los módulos deben depender poco entre sí
   - Cambios en un módulo no deben afectar otros
   - Comunicación solo a través de interfaces bien definidas

3. **Interfaces claras:**
   - Cada módulo debe tener métodos públicos bien definidos
   - Los métodos deben tener nombres descriptivos
   - Los parámetros y retornos deben ser claros

4. **Encapsulación:**
   - Detalles internos de un módulo no deben ser accesibles desde fuera
   - Solo exponer lo necesario para la comunicación

5. **Reutilización:**
   - Los módulos deben ser reutilizables
   - Deben poder usarse en diferentes contextos

---

## 🔗 Conexión con Otras Fases

### Con Fase 1 (Fundamentos):
- La arquitectura modular implementa los conceptos de RL
- El Environment es el "Entorno" del RL
- El Agent es el "Agente" del RL
- El ciclo de comunicación implementa el ciclo Estado → Acción → Recompensa

### Con Fase 2 (Entorno):
- El módulo Environment implementa todo lo de la Fase 2
- Contiene el tablero, las reglas, las recompensas

### Con Fase 3 (Agente y Visión):
- El módulo Interpreter implementa la visión limitada
- El módulo Agent implementa la toma de decisiones y el aprendizaje
- El sistema de recompensas está en el módulo Environment

### Con Fase 5 (Funcionalidades CLI):
- Los módulos deben poder usarse desde la línea de comandos
- El módulo Agent debe poder guardar/cargar modelos
- El módulo Environment debe poder activar/desactivar visualización

---

## 📝 Resumen de Módulos

| Módulo | Responsabilidades | Input | Output |
|--------|------------------|-------|--------|
| **Environment** | Gestiona tablero y reglas | Acciones del agente | Estado completo, recompensas |
| **Interpreter** | Traduce estado a visión | Estado completo | Visión limitada |
| **Agent** | Aprende y decide | Visión limitada | Acciones, modelos guardados |

---

## 🎓 Siguiente Paso

Una vez organizado el código en módulos bien definidos, estarás listo para la **Fase 5: Funcionalidades del Programa (CLI)**, donde añadirás la capacidad de ejecutar el programa desde la terminal con argumentos específicos para entrenamiento, visualización y gestión de modelos.
