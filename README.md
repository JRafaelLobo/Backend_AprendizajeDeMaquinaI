
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
