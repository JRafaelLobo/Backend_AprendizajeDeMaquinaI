
# BotMedics Backend

Este repositorio contiene el backend del proyecto **BotMedics**, un sistema de Aprendizaje de Máquina diseñado para ser modular, mantenible y seguro. Aquí encontrarás información sobre su arquitectura, configuración, flujo de ejecución y lineamientos para extender o modificar componentes sin comprometer la estabilidad del sistema.

## 🚀 Características Clave
- Arquitectura limpia y modular
- Configuración centralizada por entorno
- Scripts automatizados para despliegue y ejecución
- Entornos aislados y gestionados mediante variables de entorno

---

## ⚙️ Configuración de Entornos
El proyecto utiliza variables de entorno para garantizar seguridad y flexibilidad.

- Usa el archivo `.env.example` como plantilla.
- Copia y renombra:

```bash
cp .env.example .env
```

### Entornos disponibles
- **development** – Para trabajo local
- **staging** – Para pruebas previas a producción
- **production** – Para despliegues reales

Todas las configuraciones se administran desde `infrastructure/config`.

### Selección del modelo de embeddings
El backend admite dos modos controlados con `EMBEDDINGS_MODE`:

- `full`: usa BioSentVec y mantiene la compatibilidad con el índice FAISS actual.
- `lite`: usa `bioformers/bioformer-8L` y guarda o carga un índice FAISS separado para evitar incompatibilidades de dimensión.

Ejemplo:

```bash
EMBEDDINGS_MODE=full
```

```bash
EMBEDDINGS_MODE=lite
```

Al cambiar de modo, debes generar el índice FAISS correspondiente con ese mismo embedding. Si activas `lite` sin haber generado antes `backend/chat/FAISS/faiss_index_renal_lite/index.faiss`, el endpoint de chat responderá que el índice no está disponible.

Ejemplo para generar el índice `lite`:

```bash
cd backend
EMBEDDINGS_MODE=lite ../.venv/bin/python chat/entrenar.py
```

---

## ▶️ Ejecución del Proyecto
### Opción 1: Ejecución Automática (recomendada)
Este método instala dependencias y prepara el entorno automáticamente.

```bash
./start.sh
```

### Opción 2: Ejecución Manual
Asegúrate de tener todas las variables de entorno configuradas.

Instalar dependencias (si no las tienes):

```bash
pip install -r requirements.txt
```

Si vas a usar `EMBEDDINGS_MODE=lite`, asegúrate de instalar también las dependencias necesarias para Bioformer.

Ejecutar el servidor:

```bash
python3 manage.py runserver
```

---

## 🛠️ Mantenimiento y Extensión del Sistema
- No mezcles lógica de negocio con lógica de infraestructura
- Crea módulos desacoplados para nuevos modelos ML
- Documenta cualquier componente nuevo o cambios relevantes
---
## 👥 Autores

| Nombre | Contacto |
|--------|-----------|
| **José Lobo** | https://github.com/JRafaelLobo |
| **Marcela Tovar**  | https://github.com/MarcelaTovar |
