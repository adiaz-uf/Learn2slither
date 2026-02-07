# 🐍 Learn2Slither - Guía de Uso

Guía completa para entrenar, evaluar y ejecutar el agente de Snake con Deep Q-Learning.

---

## 📋 Requisitos Previos

### 1. Instalar dependencias

```bash
# Crear entorno virtual (recomendado)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# o
.venv\Scripts\activate     # Windows

# Instalar paquetes necesarios
pip install -r requirements.txt
```

### 2. Estructura del proyecto

```
Learn2slither/
├── src/
│   └── AI_Model/
│       ├── main.py       # Punto de entrada principal
│       ├── Agent.py      # Implementación del agente DQN
│       ├── Board.py      # Lógica del juego
│       ├── Snake.py      # Clase Snake
│       └── UI/
│           └── game_ui.py  # Interfaz gráfica
├── models/               # Modelos guardados (se crea automáticamente)
└── requirements.txt
```

---

## 🚀 Ejecución Rápida

### Entrenar un nuevo modelo (100 episodios)
```bash
python src/AI_Model/main.py --sessions 100
```

### Entrenar con visualización
```bash
python src/AI_Model/main.py --sessions 100 --visual on --fps 10
```

### Evaluar modelo entrenado (sin aprendizaje)
```bash
python src/AI_Model/main.py --dontlearn --load models/snake_10x10_100ep.keras --visual on
```

### Continuar entrenamiento de modelo existente
```bash
python src/AI_Model/main.py --sessions 500 --load models/snake_10x10_100ep.keras
```

---

## 🎛️ Argumentos Disponibles

### **Argumentos de Entrenamiento**

#### `--sessions N`
- **Descripción**: Número de episodios de entrenamiento
- **Tipo**: Entero
- **Por defecto**: `100`
- **Ejemplo**: `--sessions 1000`

#### `--load PATH`
- **Descripción**: Cargar modelo pre-entrenado desde archivo
- **Tipo**: Ruta del archivo `.keras`
- **Por defecto**: `None` (crea modelo nuevo)
- **Ejemplo**: `--load models/snake_10x10_500ep_20260207.keras`
- **Nota**: También carga automáticamente el archivo `_metadata.json` asociado

#### `--save PATH`
- **Descripción**: Guardar modelo en ruta específica
- **Tipo**: Ruta del archivo
- **Por defecto**: Auto-genera nombre con timestamp
- **Ejemplo**: `--save models/my_custom_model.keras`
- **Formato auto**: `models/snake_{size}x{size}_{episodes}ep_{timestamp}.keras`

---

### **Argumentos de Modo**

#### `--dontlearn`
- **Descripción**: Modo evaluación - ejecuta el agente sin entrenar
- **Tipo**: Flag (sin valor)
- **Uso**: Debe combinarse con `--load`
- **Ejemplo**: `--dontlearn --load models/snake_model.keras`

#### `--visual {on,off}`
- **Descripción**: Habilitar/deshabilitar interfaz visual
- **Tipo**: Opción (`on` o `off`)
- **Por defecto**: `off`
- **Comportamiento**:
  - `--visual off`: Entrenamiento rápido en terminal (sin pygame)
  - `--visual on`: Muestra juego paso a paso (más lento)
  - `--visual on --dontlearn`: Interfaz completa con lobby y stats

---

### **Argumentos de Configuración**

#### `--board-size {10,14,18}`
- **Descripción**: Tamaño del tablero (NxN)
- **Tipo**: Entero
- **Opciones**: `10`, `14`, `18`
- **Por defecto**: `10`
- **Ejemplo**: `--board-size 14`
- **Nota**: El tamaño del input de la red es `4 × board_size`

#### `--fps N`
- **Descripción**: Velocidad de visualización (frames por segundo)
- **Tipo**: Entero
- **Por defecto**: `10`
- **Ejemplo**: `--fps 5` (más lento), `--fps 30` (más rápido)
- **Nota**: Solo tiene efecto con `--visual on`

#### `--debug`
- **Descripción**: Modo debug con prints detallados
- **Tipo**: Flag (sin valor)
- **Muestra**:
  - Valores Q de la red
  - Recompensas recibidas
  - Pérdida (loss) durante entrenamiento
  - Estado del input procesado
- **Ejemplo**: `--debug`
- **Nota**: Muestra debug completo para los primeros 2 episodios

---

## 📊 Ejemplos de Uso Completos

### 1. **Entrenar desde cero (básico)**
```bash
python src/AI_Model/main.py --sessions 500
```
- Entrena 500 episodios
- Tablero 10x10
- Sin visualización (rápido)
- Guarda modelo con nombre auto-generado

---

### 2. **Entrenar con debug para diagnóstico**
```bash
python src/AI_Model/main.py --sessions 50 --debug
```
- Entrena 50 episodios
- Muestra información detallada de los primeros 2 episodios
- Útil para diagnosticar problemas de aprendizaje

---

### 3. **Entrenar con visualización lenta**
```bash
python src/AI_Model/main.py --sessions 100 --visual on --fps 5
```
- Entrena 100 episodios
- Muestra juego paso a paso
- 5 FPS (lento, para observar comportamiento)

---

### 4. **Continuar entrenamiento existente**
```bash
python src/AI_Model/main.py \
  --sessions 500 \
  --load models/snake_10x10_100ep_20260207_120000.keras
```
- Carga modelo con 100 episodios previos
- Entrena 500 episodios adicionales (total: 600)
- Los metadatos se actualizan automáticamente

---

### 5. **Entrenar en tablero grande**
```bash
python src/AI_Model/main.py \
  --sessions 1000 \
  --board-size 18 \
  --save models/snake_large_board.keras
```
- Tablero 18x18 (más difícil)
- 1000 episodios
- Nombre de archivo personalizado

---

### 6. **Evaluar modelo (sin entrenar)**
```bash
python src/AI_Model/main.py \
  --dontlearn \
  --load models/snake_10x10_500ep.keras
```
- Solo evalúa el modelo (10 partidas)
- No entrena ni actualiza pesos
- Muestra estadísticas finales en terminal

---

### 7. **Jugar con interfaz completa**
```bash
python src/AI_Model/main.py \
  --dontlearn \
  --load models/snake_10x10_1000ep.keras \
  --visual on
```
- Interfaz gráfica completa con pygame
- Lobby inicial → Juego → Pantalla de estadísticas
- El agente juega con su política entrenada
- Sin actualización de pesos

---

### 8. **Debug completo con modelo cargado**
```bash
python src/AI_Model/main.py \
  --sessions 20 \
  --load models/my_model.keras \
  --debug \
  --visual on \
  --fps 2
```
- Continúa entrenamiento con debug activo
- Visualización muy lenta (2 FPS)
- Útil para ver Q-values en tiempo real

---

## 📁 Gestión de Modelos

### Formato de archivos guardados

Cada entrenamiento genera 2 archivos:

1. **Modelo**: `snake_10x10_500ep_20260207_143022.keras`
   - Red neuronal completa (pesos + arquitectura)
   
2. **Metadatos**: `snake_10x10_500ep_20260207_143022_metadata.json`
   ```json
   {
     "board_size": 10,
     "total_episodes": 500,
     "average_score": 12.4,
     "high_score": 28,
     "final_epsilon": 0.156,
     "timestamp": "2026-02-07 14:30:22"
   }
   ```

### Cargar modelos automáticamente

El sistema carga metadatos automáticamente:
```bash
python src/AI_Model/main.py \
  --load models/snake_10x10_500ep.keras \
  --sessions 500
```
- Detecta automáticamente: `snake_10x10_500ep_metadata.json`
- Continúa desde episodio 501
- El nuevo modelo tendrá: `1000ep` en el nombre

---

## 🎯 Flujo de Trabajo Recomendado

### Fase 1: Entrenamiento inicial con debug
```bash
# Ver si el modelo aprende correctamente
python src/AI_Model/main.py --sessions 50 --debug
```

### Fase 2: Entrenamiento extensivo
```bash
# Entrenar sin visualización (más rápido)
python src/AI_Model/main.py --sessions 500
```

### Fase 3: Continuar si es necesario
```bash
# Si el score promedio < 15, continuar entrenando
python src/AI_Model/main.py \
  --sessions 500 \
  --load models/snake_10x10_500ep_*.keras
```

### Fase 4: Evaluar rendimiento
```bash
# Probar sin aprendizaje
python src/AI_Model/main.py \
  --dontlearn \
  --load models/snake_10x10_1000ep_*.keras
```

### Fase 5: Demostración visual
```bash
# Mostrar agente entrenado con interfaz
python src/AI_Model/main.py \
  --dontlearn \
  --load models/snake_10x10_1000ep_*.keras \
  --visual on \
  --fps 10
```

---

## 🐛 Troubleshooting

### El modelo no aprende (score estancado en 2-3)

**Solución 1**: Entrenar más episodios
```bash
python src/AI_Model/main.py --sessions 1000
```

**Solución 2**: Verificar con debug
```bash
python src/AI_Model/main.py --sessions 20 --debug
```
- Verifica que Q-values aumenten
- Verifica que loss disminuya
- Verifica que el agente come manzanas (reward +10)

---

### Error: "No such file or directory"
```bash
# Asegúrate de ejecutar desde la raíz del proyecto
cd Learn2slither
python src/AI_Model/main.py --sessions 100
```

---

### Error: "ModuleNotFoundError"
```bash
# Reinstala dependencias
pip install -r requirements.txt
```

---

### Visualización muy lenta
```bash
# Aumenta FPS o desactiva visualización
python src/AI_Model/main.py --sessions 100 --visual on --fps 30
# o
python src/AI_Model/main.py --sessions 100 --visual off
```

---

## 📈 Métricas de Rendimiento Esperadas

| Episodios | Score Promedio | High Score | Epsilon |
|-----------|----------------|------------|---------|
| 0-50      | 2-4            | 4-6        | 1.0-0.6 |
| 50-200    | 4-8            | 10-15      | 0.6-0.3 |
| 200-500   | 8-15           | 20-30      | 0.3-0.1 |
| 500-1000  | 15-25          | 35-50      | 0.1-0.01|
| 1000+     | 20-30          | 50-80      | 0.01    |

**Nota**: Estos valores son aproximados. El rendimiento varía según configuración y semillas aleatorias.

---

## 🎓 Configuración Avanzada

### Entrenar múltiples modelos en paralelo
```bash
# Terminal 1: Tablero pequeño
python src/AI_Model/main.py --sessions 1000 --board-size 10

# Terminal 2: Tablero mediano
python src/AI_Model/main.py --sessions 1000 --board-size 14

# Terminal 3: Tablero grande
python src/AI_Model/main.py --sessions 1500 --board-size 18
```

### Experimentar con diferentes hiperparámetros
Modifica `Agent.py` para cambiar:
- `learning_rate`: Velocidad de aprendizaje (default: 0.001)
- `gamma`: Factor de descuento (default: 0.95)
- `epsilon_decay`: Velocidad de decay de exploración (default: 0.995)

```bash
# Después de modificar Agent.py:
python src/AI_Model/main.py --sessions 500 --debug
```

---

## 📝 Notas Importantes

1. **Entrenamiento sin visualización es ~10x más rápido**
2. **Los modelos se guardan automáticamente** al finalizar
3. **Metadatos permiten continuar entrenamiento** sin pérdida de progreso
4. **Debug solo se activa en primeros 2 episodios** para evitar spam
5. **Epsilon decay automático**: No necesitas configurarlo manualmente

---

## 💡 Tips

- ✅ Entrena al menos **500 episodios** para ver aprendizaje real
- ✅ Usa `--debug` solo para diagnóstico inicial (ralentiza ejecución)
- ✅ Guarda modelos intermedios con `--save` para no perder progreso
- ✅ Evalúa con `--dontlearn` para ver performance real sin exploración
- ✅ Si el score no sube tras 1000 episodios, revisa rewards en `Board.py`

---

## 📞 Soporte

Para más información, consulta:
- `README_IMPLEMENTACION_RL.md`: Detalles de la implementación DQN
- `README_GUIA_COMPLETA.md`: Guía completa del proyecto
- Archivos de fase: `README_FASE*.md`

---

**¡Feliz entrenamiento! 🎮🐍**
