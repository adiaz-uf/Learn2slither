# Fase 5: Funcionalidades del Programa (CLI)

## 📚 Propósito de esta Fase

Esta fase añade la capacidad de ejecutar el programa desde la línea de comandos (CLI - Command Line Interface) con argumentos específicos. Esto permite controlar el entrenamiento, la visualización, la gestión de modelos y otros aspectos del programa de manera flexible y automatizada.

---

## 🎯 Conceptos Fundamentales

### ¿Qué es una Interfaz de Línea de Comandos (CLI)?

**Definición:**
Una CLI es una forma de interactuar con un programa mediante texto y comandos escritos en una terminal o consola, en lugar de usar una interfaz gráfica.

**Ventajas:**
- **Automatización:** Puedes ejecutar el programa con scripts
- **Reproducibilidad:** Mismos argumentos = mismos resultados
- **Eficiencia:** Más rápido que interfaces gráficas para tareas repetitivas
- **Flexibilidad:** Muchas opciones de configuración
- **Entrenamiento masivo:** Puedes entrenar sin intervención manual

**Papel en el proyecto:**
- Permite entrenar el agente de manera automatizada
- Facilita la experimentación con diferentes configuraciones
- Permite ejecutar entrenamientos largos sin supervisión
- Facilita la evaluación y comparación de modelos

---

## ⚙️ Funcionalidades Requeridas

### 1. Control de Sesiones

**¿Qué es?**
La capacidad de especificar cuántas partidas (sesiones/episodios) debe jugar el agente durante el entrenamiento.

**Argumento:**
- `-sessions <número>` o `--sessions <número>`
- Ejemplo: `python main.py -sessions 100`

**Funcionalidad:**
- El programa ejecuta el número especificado de partidas
- Cada partida es un episodio completo (desde inicio hasta game over)
- Después de cada partida, el entorno se reinicia
- El agente aprende de cada partida

**Casos de uso:**
- Entrenar con pocas sesiones para pruebas rápidas: `-sessions 10`
- Entrenar con muchas sesiones para modelos fuertes: `-sessions 1000`
- Entrenar modelos intermedios: `-sessions 100`

**Implementación conceptual:**
```
1. Leer argumento -sessions
2. Inicializar entorno y agente
3. Para cada sesión (1 hasta número_sesiones):
   a. Reiniciar entorno
   b. Mientras no sea game over:
      - Obtener visión del entorno
      - Agente elige acción
      - Entorno ejecuta acción
      - Agente aprende de la recompensa
   c. Registrar estadísticas de la sesión
4. Guardar modelo final (opcional)
```

**Papel en el proyecto:**
- Controla la duración del entrenamiento
- Permite generar modelos con diferentes niveles de entrenamiento
- Facilita la experimentación sistemática
- Es necesario para cumplir con los requisitos (1, 10, 100+ sesiones)

---

### 2. Guardar y Cargar Modelos

#### 2.1 Guardar Modelos

**¿Qué es?**
La capacidad de exportar el conocimiento aprendido (Q-Table o pesos de red neuronal) a un archivo de texto.

**Argumento:**
- `-save <nombre_archivo>` o `--save <nombre_archivo>`
- Ejemplo: `python main.py -sessions 100 -save model_100.txt`

**Funcionalidad:**
- Al finalizar el entrenamiento, guarda el modelo en un archivo `.txt`
- El archivo contiene toda la información necesaria para reconstruir el modelo
- Formato del archivo debe ser legible y parseable

**Contenido del archivo (para Q-Table):**
- Cada estado posible (visión) y sus valores Q para cada acción
- Puede ser formato clave-valor, tabla, o estructura similar
- Debe incluir metadatos si es necesario (parámetros de aprendizaje, etc.)

**Contenido del archivo (para DQN):**
- Pesos de la red neuronal
- Arquitectura de la red
- Parámetros de entrenamiento

**Casos de uso:**
- Guardar modelo después de entrenamiento: `-save model_trained.txt`
- Guardar modelos en diferentes etapas: `-save model_10.txt`, `-save model_100.txt`
- Guardar modelos para evaluación posterior

**Papel en el proyecto:**
- Permite preservar el conocimiento aprendido
- Facilita la evaluación de modelos
- Permite continuar el entrenamiento más tarde
- Es necesario para cumplir con los requisitos de entrega

---

#### 2.2 Cargar Modelos

**¿Qué es?**
La capacidad de cargar un modelo previamente entrenado desde un archivo.

**Argumento:**
- `-load <nombre_archivo>` o `--load <nombre_archivo>`
- Ejemplo: `python main.py -load model_100.txt`

**Funcionalidad:**
- Lee el archivo especificado
- Reconstruye la Q-Table o red neuronal con los valores guardados
- El agente comienza con el conocimiento del modelo cargado

**Casos de uso:**
- Continuar entrenando un modelo existente: `-load model_50.txt -sessions 50`
- Evaluar un modelo sin entrenar: `-load model_100.txt -dontlearn`
- Probar diferentes modelos: `-load model_A.txt`, `-load model_B.txt`

**Comportamiento:**
- Si se carga un modelo y se especifica `-sessions`, el agente puede:
  - Continuar aprendiendo (si no se usa `-dontlearn`)
  - Solo jugar sin aprender (si se usa `-dontlearn`)

**Papel en el proyecto:**
- Permite reutilizar modelos entrenados
- Facilita la evaluación y comparación
- Permite entrenamiento incremental
- Es necesario para cumplir con los requisitos

---

### 3. Modo "No Aprender" (Explotación)

**¿Qué es?**
Un modo donde el agente usa su conocimiento actual pero no actualiza su función Q. Es pura explotación sin exploración ni aprendizaje.

**Argumento:**
- `-dontlearn` o `--dontlearn`
- Ejemplo: `python main.py -load model_100.txt -dontlearn`

**Funcionalidad:**
- El agente consulta su función Q para tomar decisiones
- El agente NO actualiza su función Q después de cada acción
- El agente NO explora (solo usa lo que sabe)
- El agente juega de manera "óptima" según su conocimiento actual

**Comportamiento:**
- El agente siempre elige la acción con mayor valor Q
- No hay aprendizaje, no hay actualizaciones
- Útil para evaluar el rendimiento puro del modelo
- Útil para demostrar qué ha aprendido el agente

**Casos de uso:**
- Evaluar modelo entrenado: `-load model_100.txt -dontlearn`
- Ver cómo juega el agente sin modificar el modelo
- Comparar rendimiento de diferentes modelos
- Demostración del aprendizaje del agente

**Diferencia con modo normal:**
- **Modo normal:** Agente aprende y actualiza Q-Table
- **Modo -dontlearn:** Agente solo usa Q-Table, no la modifica

**Papel en el proyecto:**
- Permite evaluar el rendimiento sin sesgar el modelo
- Facilita la demostración del aprendizaje
- Es necesario para cumplir con los requisitos de evaluación
- Permite comparar modelos de manera justa

---

### 4. Control de Visualización

#### 4.1 Desactivar Gráficos

**¿Qué es?**
La capacidad de ejecutar el programa sin mostrar la interfaz gráfica.

**Argumento:**
- `-visual off` o `--visual off`
- Ejemplo: `python main.py -sessions 1000 -visual off`

**Funcionalidad:**
- El programa ejecuta sin abrir ventanas gráficas
- El entrenamiento es mucho más rápido
- Útil para entrenamientos largos o masivos

**Ventajas:**
- **Velocidad:** Sin renderizado gráfico, el programa es mucho más rápido
- **Recursos:** Usa menos memoria y CPU
- **Automatización:** Puede ejecutarse en servidores sin interfaz gráfica
- **Eficiencia:** Permite entrenar miles de sesiones rápidamente

**Casos de uso:**
- Entrenamiento masivo: `-sessions 10000 -visual off`
- Entrenamiento en servidor sin GUI
- Pruebas rápidas sin necesidad de ver el juego

**Comportamiento:**
- El entorno sigue funcionando normalmente
- Las reglas del juego se mantienen
- Solo se omite el renderizado visual
- Puede seguir mostrando información en terminal (estadísticas, progreso)

**Papel en el proyecto:**
- Permite entrenamiento eficiente
- Facilita la experimentación rápida
- Es necesario para entrenamientos largos
- Permite usar el programa en diferentes entornos

---

#### 4.2 Control de Velocidad

**¿Qué es?**
La capacidad de controlar qué tan rápido se ejecuta el juego visualmente.

**Argumento:**
- `-speed <valor>` o `--speed <valor>`
- Ejemplo: `python main.py -speed 10`

**Funcionalidad:**
- Controla los FPS (frames por segundo) del juego
- Valores más altos = juego más rápido
- Valores más bajos = juego más lento

**Casos de uso:**
- Ver el juego en cámara lenta: `-speed 5`
- Ver el juego a velocidad normal: `-speed 20`
- Ver el juego muy rápido: `-speed 60`

**Implementación:**
- Controla el `clock.tick()` de Pygame
- Afecta solo la visualización, no la lógica del juego
- No afecta el aprendizaje (el agente aprende igual)

**Papel en el proyecto:**
- Facilita la observación del comportamiento del agente
- Permite ajustar la visualización según necesidades
- Útil para debugging y demostración

---

#### 4.3 Modo Paso a Paso

**¿Qué es?**
Un modo donde el juego avanza solo cuando el usuario presiona una tecla.

**Argumento:**
- `-step-by-step` o `--step-by-step`
- Ejemplo: `python main.py -load model_100.txt -step-by-step`

**Funcionalidad:**
- Después de cada acción, el juego se pausa
- El usuario debe presionar una tecla para continuar
- Permite observar cada decisión del agente en detalle

**Casos de uso:**
- Analizar decisiones del agente paso a paso
- Debugging detallado
- Entender el comportamiento del agente
- Demostración educativa

**Comportamiento:**
1. Agente toma una acción
2. Entorno ejecuta la acción
3. Juego se pausa y muestra el estado
4. Usuario presiona tecla (ej: espacio)
5. Juego continúa con la siguiente acción

**Papel en el proyecto:**
- Facilita el análisis detallado
- Útil para debugging
- Permite entender el proceso de decisión
- Útil para demostraciones educativas

---

## 🔧 Parsing de Argumentos

### ¿Qué es el Parsing de Argumentos?

**Definición:**
El proceso de leer y procesar los argumentos proporcionados en la línea de comandos.

**Herramientas comunes en Python:**
- `argparse`: Módulo estándar de Python para parsing de argumentos
- `sys.argv`: Lista simple de argumentos (más básico)
- Librerías de terceros (click, etc.)

**Ejemplo conceptual:**
```python
# Usuario ejecuta:
python main.py -sessions 100 -save model.txt -visual off

# El programa debe:
1. Detectar -sessions y leer el valor 100
2. Detectar -save y leer el valor model.txt
3. Detectar -visual y leer el valor off
4. Configurar el programa según estos argumentos
```

**Papel en el proyecto:**
- Permite que el programa responda a diferentes configuraciones
- Facilita la automatización
- Hace el programa más flexible y usable

---

## 📋 Ejemplos de Uso

### Entrenamiento Básico
```
python main.py -sessions 100
```
- Entrena el agente por 100 sesiones
- Muestra gráficos (por defecto)
- No guarda modelo (a menos que se especifique)

### Entrenamiento y Guardar
```
python main.py -sessions 100 -save model_100.txt
```
- Entrena por 100 sesiones
- Guarda el modelo al finalizar

### Entrenamiento Rápido (Sin Gráficos)
```
python main.py -sessions 1000 -visual off -save model_1000.txt
```
- Entrena por 1000 sesiones
- Sin gráficos (más rápido)
- Guarda el modelo

### Cargar y Continuar Entrenando
```
python main.py -load model_50.txt -sessions 50 -save model_100.txt
```
- Carga modelo con 50 sesiones
- Entrena 50 sesiones más
- Guarda modelo con 100 sesiones totales

### Evaluar Modelo (Sin Aprender)
```
python main.py -load model_100.txt -dontlearn -sessions 10
```
- Carga modelo entrenado
- Juega 10 sesiones sin aprender
- Muestra el rendimiento puro del modelo

### Análisis Paso a Paso
```
python main.py -load model_100.txt -dontlearn -step-by-step
```
- Carga modelo
- Juega paso a paso
- Permite analizar cada decisión

---

## 🔗 Integración con Módulos

### Conexión con Módulo Environment

**Visualización:**
- El módulo Environment debe poder activar/desactivar el renderizado
- Debe respetar el argumento `-visual off`
- Debe controlar la velocidad según `-speed`

**Paso a paso:**
- El módulo Environment debe pausar después de cada acción
- Debe esperar input del usuario en modo `-step-by-step`

---

### Conexión con Módulo Agent

**Guardar/Cargar:**
- El módulo Agent debe implementar `save_model(filename)`
- El módulo Agent debe implementar `load_model(filename)`
- Debe manejar el formato de archivo `.txt`

**Modo No Aprender:**
- El módulo Agent debe poder desactivar el aprendizaje
- Cuando `-dontlearn` está activo, no debe actualizar la Q-Table
- Debe seguir tomando decisiones basadas en la Q-Table

---

### Conexión con Módulo Principal

**Orquestación:**
- El módulo principal (main.py) debe parsear los argumentos
- Debe configurar los módulos según los argumentos
- Debe ejecutar el ciclo de entrenamiento/juego
- Debe manejar el guardado de modelos al finalizar

---

## 📝 Resumen de Argumentos

| Argumento | Descripción | Ejemplo |
|-----------|-------------|---------|
| `-sessions <n>` | Número de partidas a entrenar | `-sessions 100` |
| `-save <file>` | Guardar modelo al finalizar | `-save model.txt` |
| `-load <file>` | Cargar modelo existente | `-load model.txt` |
| `-dontlearn` | Desactivar aprendizaje | `-dontlearn` |
| `-visual off` | Desactivar gráficos | `-visual off` |
| `-speed <n>` | Controlar velocidad (FPS) | `-speed 20` |
| `-step-by-step` | Modo paso a paso | `-step-by-step` |

---

## 🎓 Siguiente Paso

Una vez implementadas todas las funcionalidades CLI, estarás listo para la **Fase 6: Entrenamiento y Entrega**, donde entrenarás modelos con diferentes niveles de entrenamiento y los organizarás para la entrega final.
