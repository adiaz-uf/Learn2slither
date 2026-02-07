# Fase 6: Entrenamiento y Entrega

## 📚 Propósito de esta Fase

Esta fase se enfoca en entrenar modelos con diferentes niveles de experiencia, verificar que cumplen con los objetivos, y organizar todo para la entrega final. Es la fase donde todo el trabajo anterior se consolida en modelos funcionales y demostrables.

---

## 🎯 Objetivo Final

### Meta del Proyecto

**Objetivo principal:**
La serpiente debe lograr una longitud de al menos **10 segmentos** durante el juego.

**¿Por qué es importante?**
- Demuestra que el agente ha aprendido una estrategia efectiva
- Indica que el agente puede navegar el tablero sin chocar
- Muestra que el agente puede encontrar y comer comida consistentemente
- Es un indicador claro de aprendizaje exitoso

**Cómo se mide:**
- Durante una partida, la serpiente debe alcanzar longitud ≥ 10
- Esto debe ocurrir de manera consistente (no solo por casualidad)
- El modelo debe poder lograrlo cuando se carga y juega

**Papel en el proyecto:**
- Es el criterio de éxito principal
- Guía el proceso de entrenamiento
- Determina si un modelo está listo para entrega
- Es lo que se evaluará en la prueba final

---

## 🏋️ Generar Modelos con Diferentes Niveles de Entrenamiento

### ¿Por qué Múltiples Modelos?

**Razones:**
1. **Demostración de progreso:** Muestra cómo el agente mejora con más entrenamiento
2. **Comparación:** Permite ver la diferencia entre poco y mucho entrenamiento
3. **Evaluación:** Facilita la evaluación del proyecto
4. **Requisitos:** Es un requisito explícito del proyecto

**Papel en el proyecto:**
- Muestra el proceso de aprendizaje
- Facilita la evaluación y demostración
- Cumple con los requisitos de entrega

---

### 1. Modelo Poco Entrenado (1 Sesión)

**Características:**
- Entrenado con solo **1 sesión** (1 partida completa)
- El agente apenas ha comenzado a aprender
- Tiene conocimiento mínimo del entorno

**Comportamiento esperado:**
- El agente juega de manera casi aleatoria
- Puede chocar rápidamente
- No muestra estrategias claras
- Longitud típica: 3-5 segmentos (casi no crece)

**Cómo generarlo:**
```
python main.py -sessions 1 -save models/model_1_session.txt
```

**Propósito:**
- Muestra el estado inicial del agente
- Demuestra que el agente puede aprender (comparado con modelos más entrenados)
- Sirve como línea base para comparación

**Archivo resultante:**
- `models/model_1_session.txt`
- Contiene la Q-Table con muy pocos valores aprendidos
- La mayoría de estados tienen valores Q iniciales (cero o aleatorios)

---

### 2. Modelo Medio Entrenado (10 Sesiones)

**Características:**
- Entrenado con **10 sesiones** (10 partidas completas)
- El agente ha tenido oportunidad de explorar y aprender
- Tiene conocimiento intermedio del entorno

**Comportamiento esperado:**
- El agente muestra algunos patrones de comportamiento
- Puede evitar algunas colisiones obvias
- Puede buscar comida ocasionalmente
- Longitud típica: 5-8 segmentos

**Cómo generarlo:**
```
python main.py -sessions 10 -save models/model_10_sessions.txt
```

**Propósito:**
- Muestra el progreso del aprendizaje
- Demuestra que el agente está mejorando
- Sirve como punto intermedio entre poco y mucho entrenamiento

**Archivo resultante:**
- `models/model_10_sessions.txt`
- Contiene más valores Q aprendidos que el modelo de 1 sesión
- Muestra patrones de aprendizaje pero aún incompletos

---

### 3. Modelo Muy Entrenado (100+ Sesiones)

**Características:**
- Entrenado con **100 o más sesiones** (100+ partidas completas)
- El agente ha tenido amplia experiencia
- Tiene conocimiento sustancial del entorno

**Comportamiento esperado:**
- El agente juega de manera inteligente
- Evita colisiones consistentemente
- Busca y come comida de manera eficiente
- Puede alcanzar longitudes altas (10+ segmentos)
- Muestra estrategias claras y efectivas

**Cómo generarlo:**
```
python main.py -sessions 100 -save models/model_100_sessions.txt
```

**O con más sesiones:**
```
python main.py -sessions 500 -save models/model_500_sessions.txt
```

**Propósito:**
- Demuestra el potencial completo del aprendizaje
- Muestra que el agente puede lograr el objetivo (longitud ≥ 10)
- Es el modelo principal para evaluación y demostración

**Archivo resultante:**
- `models/model_100_sessions.txt` (o más)
- Contiene valores Q bien aprendidos
- Muestra estrategias óptimas o cercanas a óptimas

**Tiempo de entrenamiento:**
- Puede tomar tiempo significativo (depende de la implementación)
- Considera usar `-visual off` para acelerar
- Puede ejecutarse en segundo plano o durante la noche

---

## 📁 Organización del Repositorio

### Estructura Requerida

**Estructura de carpetas:**
```
Learn2Slither/
├── src/
│   ├── Environment/
│   │   └── game_board.py
│   ├── Interpreter/
│   │   └── vision.py
│   ├── Agent/
│   │   ├── q_learning.py
│   │   └── model_io.py
│   └── main.py
├── models/
│   ├── model_1_session.txt
│   ├── model_10_sessions.txt
│   └── model_100_sessions.txt
├── README.md
└── requirements.txt
```

**Componentes:**

1. **Código fuente (`src/`):**
   - Todo el código del proyecto
   - Módulos Environment, Interpreter, Agent
   - Archivo principal (main.py)
   - Cualquier otro código auxiliar

2. **Carpeta `models/`:**
   - Contiene todos los archivos de modelos entrenados
   - Formato: archivos `.txt` con los modelos
   - Nombres descriptivos que indican el nivel de entrenamiento

3. **Documentación:**
   - README.md con instrucciones
   - Puede incluir documentación adicional
   - Puede incluir ejemplos de uso

4. **Dependencias:**
   - requirements.txt con las librerías necesarias
   - Facilita la instalación del proyecto

**Papel en el proyecto:**
- Facilita la evaluación
- Muestra organización profesional
- Permite reproducibilidad
- Cumple con los requisitos de entrega

---

### Nombres de Archivos de Modelos

**Convención recomendada:**
- `model_1_session.txt` - Modelo con 1 sesión
- `model_10_sessions.txt` - Modelo con 10 sesiones
- `model_100_sessions.txt` - Modelo con 100 sesiones
- `model_500_sessions.txt` - Modelo con 500 sesiones (opcional)

**Ventajas:**
- Nombres descriptivos y claros
- Fácil de identificar el nivel de entrenamiento
- Consistente y profesional

**Alternativas:**
- `model_basic.txt`, `model_intermediate.txt`, `model_advanced.txt`
- `model_1.txt`, `model_10.txt`, `model_100.txt`
- Cualquier convención clara y consistente

---

## ✅ Prueba Final

### ¿Qué es la Prueba Final?

**Definición:**
Una verificación completa de que el modelo entrenado funciona correctamente y cumple con los objetivos.

**Componentes de la prueba:**

1. **Cargar el modelo:**
   - Verificar que el modelo se carga correctamente
   - Verificar que no hay errores al cargar
   - Verificar que el agente puede usar el modelo cargado

2. **Jugar sin aprender:**
   - Cargar el modelo con `-load` y `-dontlearn`
   - Verificar que el agente juega usando el conocimiento aprendido
   - Verificar que el agente NO aprende (no modifica el modelo)

3. **Verificar comportamiento inteligente:**
   - El agente no debe chocar inmediatamente
   - El agente debe mostrar estrategias claras
   - El agente debe buscar comida
   - El agente debe evitar colisiones obvias

4. **Verificar objetivo (longitud ≥ 10):**
   - El agente debe poder alcanzar longitud 10 o más
   - Esto debe ocurrir de manera consistente
   - No debe ser solo por casualidad

**Cómo realizar la prueba:**
```
# 1. Cargar modelo de 100 sesiones
python main.py -load models/model_100_sessions.txt -dontlearn

# 2. Observar el comportamiento
# - ¿Choca inmediatamente? (No debería)
# - ¿Busca comida? (Sí debería)
# - ¿Evita colisiones? (Sí debería)
# - ¿Alcanza longitud 10+? (Sí debería)

# 3. Verificar estadísticas
# - Longitud máxima alcanzada
# - Número de colisiones
# - Eficiencia en encontrar comida
```

**Criterios de éxito:**
- ✅ Modelo se carga sin errores
- ✅ Agente juega de manera inteligente (no aleatoria)
- ✅ Agente no choca inmediatamente
- ✅ Agente alcanza longitud ≥ 10 consistentemente
- ✅ Agente muestra estrategias claras

**Papel en el proyecto:**
- Verifica que todo funciona correctamente
- Demuestra que el aprendizaje fue exitoso
- Es necesario antes de la entrega
- Proporciona confianza en el proyecto

---

## 📊 Proceso de Entrenamiento

### Estrategia Recomendada

**Fase 1: Desarrollo y Pruebas**
1. Entrenar con pocas sesiones (1-10) para verificar que todo funciona
2. Verificar que el agente aprende (valores Q cambian)
3. Verificar que no hay errores en el código
4. Ajustar parámetros si es necesario (tasa de aprendizaje, recompensas, etc.)

**Fase 2: Entrenamiento Incremental**
1. Entrenar modelo de 1 sesión y guardarlo
2. Entrenar modelo de 10 sesiones y guardarlo
3. Entrenar modelo de 100 sesiones y guardarlo
4. Opcionalmente, entrenar modelos con más sesiones (200, 500, 1000)

**Fase 3: Optimización**
1. Si el modelo de 100 sesiones no alcanza longitud 10:
   - Ajustar sistema de recompensas
   - Ajustar parámetros de aprendizaje (α, γ, ε)
   - Entrenar con más sesiones
   - Verificar que la visión y acciones están correctas

**Fase 4: Verificación**
1. Cargar cada modelo y verificar que funciona
2. Probar el modelo de 100 sesiones extensivamente
3. Verificar que alcanza longitud ≥ 10
4. Documentar cualquier problema o ajuste necesario

---

### Parámetros a Ajustar

**Si el agente no aprende bien:**

1. **Sistema de recompensas:**
   - Aumentar recompensa por comer manzana verde
   - Aumentar penalización por chocar
   - Ajustar recompensa por moverse sin comer

2. **Tasa de aprendizaje (α):**
   - Si es muy baja: El agente aprende muy lento
   - Si es muy alta: El agente puede ser inestable
   - Valores típicos: 0.1 a 0.5

3. **Factor de descuento (γ):**
   - Controla qué tan importante es el futuro
   - Valores típicos: 0.9 a 0.99
   - Más alto = agente valora más recompensas futuras

4. **Exploración (ε):**
   - Controla cuánto explora vs explota
   - Puede disminuir con el tiempo (ε-decay)
   - Valores típicos: 0.1 a 0.3 (después de calentamiento)

---

## 🔍 Verificación de Calidad

### Checklist Antes de Entrega

**Código:**
- ✅ Código está organizado en módulos (Environment, Interpreter, Agent)
- ✅ Código está comentado y es legible
- ✅ No hay errores de sintaxis
- ✅ El programa se ejecuta sin errores

**Funcionalidades:**
- ✅ Argumento `-sessions` funciona
- ✅ Argumento `-save` funciona
- ✅ Argumento `-load` funciona
- ✅ Argumento `-dontlearn` funciona
- ✅ Argumento `-visual off` funciona
- ✅ Argumento `-step-by-step` funciona (si se implementó)

**Modelos:**
- ✅ Modelo de 1 sesión existe y se puede cargar
- ✅ Modelo de 10 sesiones existe y se puede cargar
- ✅ Modelo de 100+ sesiones existe y se puede cargar
- ✅ Todos los modelos están en la carpeta `models/`

**Objetivo:**
- ✅ Modelo de 100 sesiones alcanza longitud ≥ 10
- ✅ El agente juega de manera inteligente
- ✅ El agente no choca inmediatamente

**Documentación:**
- ✅ README.md explica cómo usar el programa
- ✅ README.md explica cómo entrenar modelos
- ✅ README.md explica cómo cargar y evaluar modelos

---

## 📝 Resumen del Proceso

**Pasos para completar esta fase:**

1. **Entrenar modelos:**
   - Modelo de 1 sesión
   - Modelo de 10 sesiones
   - Modelo de 100+ sesiones

2. **Organizar repositorio:**
   - Código fuente en `src/`
   - Modelos en `models/`
   - Documentación actualizada

3. **Verificar funcionamiento:**
   - Cargar cada modelo
   - Verificar que juegan correctamente
   - Verificar que alcanzan longitud ≥ 10

4. **Preparar entrega:**
   - Verificar checklist
   - Documentar cualquier cosa especial
   - Asegurar que todo está completo

---

## 🎓 Siguiente Paso

Una vez completada esta fase y verificada la prueba final, el proyecto estará listo para entrega. Si todo funciona correctamente, puedes considerar la **Fase 7: Bonus (Opcional)** para mejorar aún más el proyecto con características adicionales.
