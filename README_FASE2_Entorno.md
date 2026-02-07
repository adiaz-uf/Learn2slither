# Fase 2: Construcción del Entorno (El Tablero)

## 📚 Propósito de esta Fase

En esta fase construyes el "mundo" donde vivirá y aprenderá la serpiente. El entorno es fundamental porque define todas las reglas del juego, gestiona el estado actual y proporciona las recompensas que guiarán el aprendizaje del agente.

---

## 🎯 Conceptos Fundamentales

### ¿Qué es un Entorno en RL?

**Definición:**
El entorno es el sistema con el que el agente interactúa. Define:
- El espacio de estados posibles
- Las acciones permitidas
- Las reglas de transición entre estados
- El sistema de recompensas
- Las condiciones de terminación

**En Learn2Slither:**
El entorno es el tablero de juego completo con todas sus reglas, elementos y mecánicas.

**Papel en el proyecto:**
- Es la "realidad" con la que el agente interactúa
- Proporciona la información que el agente necesita para tomar decisiones
- Ejecuta las acciones del agente y calcula las consecuencias
- Determina cuándo termina un episodio de entrenamiento

---

## 🏗️ Componentes del Entorno

### 1. Grid del Tablero (Cuadrícula)

**¿Qué es?**
Una cuadrícula de 10x10 celdas que representa el espacio de juego.

**Características:**
- **Dimensiones:** 10 filas × 10 columnas = 100 celdas totales
- **Coordenadas:** Cada celda tiene una posición única (fila, columna)
- **Bordes:** Las celdas en los bordes están limitadas por las paredes

**Representación conceptual:**
```
[0,0] [0,1] [0,2] ... [0,9]
[1,0] [1,1] [1,2] ... [1,9]
 ...
[9,0] [9,1] [9,2] ... [9,9]
```

**Papel en el proyecto:**
- Define el espacio de juego
- Establece límites físicos (paredes)
- Permite posicionar elementos (serpiente, comida)
- Facilita la detección de colisiones

**Consideraciones de diseño:**
- Debe ser fácil de representar en memoria (matriz 2D)
- Debe permitir acceso rápido a cualquier celda
- Debe validar que las coordenadas estén dentro de los límites

---

### 2. Elementos del Juego

#### 2.1 La Serpiente

**¿Qué es?**
La entidad controlada por el agente. Es una secuencia de segmentos conectados que se mueve por el tablero.

**Características iniciales:**
- **Longitud inicial:** 3 segmentos
- **Posición inicial:** Aleatoria (no fija)
- **Cabeza:** El primer segmento, desde donde se toma la decisión
- **Cuerpo/Cola:** Los segmentos restantes que siguen a la cabeza

**Representación:**
- Puede ser una lista de coordenadas: `[(x1,y1), (x2,y2), (x3,y3)]`
- El primer elemento es la cabeza
- Los siguientes son el cuerpo en orden

**Mecánica de movimiento:**
1. La cabeza se mueve a una nueva posición
2. Cada segmento del cuerpo toma la posición del segmento anterior
3. Si come comida verde, se añade un nuevo segmento
4. Si come comida roja, se elimina un segmento

**Papel en el proyecto:**
- Es el agente visual (lo que el usuario ve)
- Su cabeza es el punto de referencia para la visión del agente
- Su longitud determina la dificultad y el objetivo del juego
- Su posición y forma afectan las decisiones del agente

**Reglas importantes:**
- No puede moverse fuera de los límites del tablero
- No puede chocar con su propio cuerpo
- Debe mantener al menos longitud 1 (si llega a 0, game over)

---

#### 2.2 Manzanas Verdes (2 manzanas)

**¿Qué es?**
Objetivos positivos que la serpiente debe comer para crecer.

**Características:**
- **Cantidad:** Siempre hay exactamente 2 manzanas verdes en el tablero
- **Efecto:** Aumentan la longitud de la serpiente en +1
- **Recompensa:** Proporcionan recompensa positiva al agente
- **Reposición:** Cuando se come una, aparece una nueva en posición aleatoria

**Posicionamiento:**
- Deben aparecer en celdas vacías (no donde está la serpiente)
- Deben estar dentro de los límites del tablero
- Deben ser accesibles (no bloqueadas por la serpiente)

**Papel en el proyecto:**
- Son los objetivos principales del agente
- Proporcionan feedback positivo (recompensas)
- Aumentan la dificultad al hacer la serpiente más larga
- Guían el aprendizaje hacia comportamientos beneficiosos

**Mecánica:**
- Cuando la cabeza de la serpiente ocupa la misma celda que una manzana verde
- La serpiente crece (se añade un segmento)
- La manzana desaparece y aparece una nueva
- El agente recibe recompensa positiva

---

#### 2.3 Manzana Roja (1 manzana)

**¿Qué es?**
Un elemento peligroso que la serpiente debe evitar.

**Características:**
- **Cantidad:** Siempre hay exactamente 1 manzana roja en el tablero
- **Efecto:** Disminuye la longitud de la serpiente en -1
- **Recompensa:** Proporciona recompensa negativa al agente
- **Reposición:** Cuando se come una, aparece una nueva en posición aleatoria

**Posicionamiento:**
- Similar a las manzanas verdes (celdas vacías, dentro de límites)
- Puede aparecer cerca o lejos de la serpiente

**Papel en el proyecto:**
- Introduce riesgo y complejidad al juego
- Enseña al agente a distinguir entre elementos buenos y malos
- Proporciona feedback negativo para desalentar comportamientos
- Aumenta la dificultad estratégica

**Mecánica:**
- Cuando la cabeza de la serpiente ocupa la misma celda que la manzana roja
- La serpiente se encoge (se elimina un segmento)
- La manzana desaparece y aparece una nueva
- El agente recibe recompensa negativa
- Si la serpiente llega a longitud 0, es game over

**Estrategia del agente:**
- El agente debe aprender a evitar las manzanas rojas
- Debe priorizar las manzanas verdes sobre las rojas
- Debe evaluar si el riesgo de comer una roja vale la pena

---

### 3. Reglas de "Game Over"

El entorno debe detectar cuándo termina un episodio de juego. Hay tres condiciones que causan game over:

#### 3.1 Chocar con una Pared

**¿Qué es?**
Cuando la serpiente intenta moverse fuera de los límites del tablero.

**Detección:**
- La cabeza intenta ocupar una celda fuera del rango [0-9] × [0-9]
- O la cabeza intenta moverse a coordenadas negativas o mayores que 9

**Papel en el proyecto:**
- Enseña al agente a respetar los límites del tablero
- Proporciona feedback negativo inmediato
- Termina el episodio para comenzar uno nuevo

**Consideraciones:**
- Debe detectarse antes de actualizar la posición
- Debe proporcionar recompensa negativa clara
- Debe reiniciar el entorno para el próximo episodio

---

#### 3.2 Chocar con su Propia Cola

**¿Qué es?**
Cuando la cabeza de la serpiente intenta ocupar una celda que ya contiene parte de su cuerpo.

**Detección:**
- La nueva posición de la cabeza coincide con cualquier segmento del cuerpo
- No cuenta si solo hay 1 segmento (no puede chocar consigo misma)

**Papel en el proyecto:**
- Enseña al agente a evitar movimientos que causen colisiones
- Se vuelve más relevante cuando la serpiente es larga
- Aumenta la dificultad estratégica

**Consideraciones:**
- Debe verificar todos los segmentos excepto la cabeza
- Debe proporcionar recompensa negativa
- Es una condición común cuando la serpiente es larga

---

#### 3.3 Longitud de Serpiente Llega a 0

**¿Qué es?**
Cuando la serpiente come tantas manzanas rojas que su longitud se reduce a cero.

**Detección:**
- Después de comer una manzana roja, la longitud es 0
- O la longitud se vuelve negativa (aunque esto no debería pasar)

**Papel en el proyecto:**
- Enseña al agente a evitar comer manzanas rojas cuando está cerca del límite
- Añade una condición de riesgo adicional
- Proporciona feedback negativo por malas decisiones acumuladas

**Consideraciones:**
- Debe verificarse después de cada acción que afecte la longitud
- Debe ser una condición de game over clara
- El agente debe aprender a mantener longitud mínima

---

### 4. Interfaz Gráfica

**¿Qué es?**
Una representación visual del estado del tablero en tiempo real.

**Componentes visuales:**
- **Tablero:** Cuadrícula visible con celdas
- **Serpiente:** Representación visual de los segmentos (color, forma)
- **Manzanas verdes:** Elementos visuales distintivos
- **Manzana roja:** Elemento visual distintivo
- **Información:** Puntuación, longitud, etc.

**Características:**
- Debe actualizarse en tiempo real
- Debe ser clara y fácil de interpretar
- Debe mostrar el estado actual del juego

**Papel en el proyecto:**
- Permite visualizar el comportamiento del agente
- Facilita el debugging y la comprensión
- Hace el proyecto más interactivo y comprensible
- Permite verificar que el agente está aprendiendo

**Consideraciones de diseño:**
- Debe ser opcional (poder desactivarse para entrenamiento rápido)
- Debe tener control de velocidad (FPS)
- Debe ser clara y no distraer del aprendizaje
- Puede incluir información adicional (estadísticas, recompensas)

**Modos de visualización:**
- **Activada:** Muestra gráficos en tiempo real (más lento, mejor para debugging)
- **Desactivada:** Sin gráficos (más rápido, mejor para entrenamiento masivo)

---

## 🔄 Flujo de Funcionamiento del Entorno

### Ciclo Principal del Entorno

1. **Inicialización:**
   - Crear tablero vacío
   - Colocar serpiente en posición aleatoria (longitud 3)
   - Colocar 2 manzanas verdes en posiciones aleatorias
   - Colocar 1 manzana roja en posición aleatoria

2. **Bucle de Juego:**
   - Recibir acción del agente
   - Validar que la acción es legal
   - Mover la serpiente según la acción
   - Verificar colisiones (paredes, cuerpo propio)
   - Verificar si comió comida (verde o roja)
   - Actualizar longitud de serpiente
   - Reposicionar comida si fue comida
   - Calcular recompensa
   - Verificar condiciones de game over
   - Actualizar visualización (si está activada)
   - Retornar: (nuevo_estado, recompensa, game_over, info)

3. **Reset (Nuevo Episodio):**
   - Limpiar tablero
   - Reinicializar serpiente
   - Reposicionar toda la comida
   - Resetear contadores

---

## 🎮 Responsabilidades del Entorno

### Lo que el Entorno DEBE hacer:

1. **Gestionar el Estado:**
   - Mantener la posición de todos los elementos
   - Actualizar el estado después de cada acción
   - Proporcionar el estado al agente cuando lo solicite

2. **Validar Acciones:**
   - Verificar que las acciones son legales
   - Manejar acciones inválidas apropiadamente

3. **Calcular Recompensas:**
   - Asignar recompensas positivas por comer manzanas verdes
   - Asignar recompensas negativas por comer manzanas rojas
   - Asignar recompensas negativas por colisiones
   - Asignar recompensas por movimientos (puede ser neutra o ligeramente negativa)

4. **Detectar Terminación:**
   - Identificar condiciones de game over
   - Señalar cuando termina un episodio

5. **Gestionar Elementos:**
   - Reposicionar comida cuando es comida
   - Asegurar que la comida no aparezca sobre la serpiente
   - Mantener el número correcto de cada tipo de comida

### Lo que el Entorno NO debe hacer:

1. **Tomar Decisiones:**
   - El entorno no decide qué acción tomar
   - Solo ejecuta las acciones del agente

2. **Aprender:**
   - El entorno no tiene memoria o aprendizaje
   - Es determinista (mismas acciones → mismos resultados)

3. **Proporcionar Información Extra:**
   - No debe dar información global del tablero al agente
   - Solo proporciona lo que la serpiente puede "ver" (Fase 3)

---

## 🔗 Conexión con Otras Fases

### Con Fase 1 (Fundamentos):
- El entorno implementa el concepto de "Entorno" del RL
- Proporciona estados, ejecuta acciones, calcula recompensas
- Sigue el ciclo: Estado → Acción → Recompensa → Nuevo Estado

### Con Fase 3 (Agente y Visión):
- El entorno proporciona el estado completo del tablero
- El Interpreter (Fase 3) traduce este estado a la visión limitada del agente
- El entorno recibe acciones del agente y las ejecuta

### Con Fase 4 (Estructura Modular):
- El entorno es el módulo "Environment"
- Se comunica con el módulo "Interpreter" y "Agent"
- Sigue el flujo: Environment → State → Interpreter → Agent → Action → Environment

---

## 📝 Resumen de Componentes

| Componente | Características | Papel en el Proyecto |
|------------|----------------|---------------------|
| **Grid 10×10** | 100 celdas, coordenadas (0-9, 0-9) | Define el espacio de juego |
| **Serpiente** | Longitud 3 inicial, posición aleatoria | Entidad controlada por el agente |
| **Manzanas Verdes (2)** | Aumentan longitud +1, recompensa positiva | Objetivos principales |
| **Manzana Roja (1)** | Disminuye longitud -1, recompensa negativa | Elemento de riesgo |
| **Reglas Game Over** | 3 condiciones: pared, cola, longitud 0 | Define terminación de episodios |
| **Interfaz Gráfica** | Visualización en tiempo real | Facilita debugging y comprensión |

---

## 🎓 Siguiente Paso

Una vez construido el entorno completo, estarás listo para la **Fase 3: El Agente y la Visión**, donde conectarás el entorno con el sistema de aprendizaje, definiendo cómo el agente "ve" el mundo y cómo toma decisiones.
