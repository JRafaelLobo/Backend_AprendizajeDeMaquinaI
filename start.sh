#!/usr/bin/env bash
set -e  # Corta si hay un error

echo "=== Creando entorno virtual ==="
python3 -m venv venv

echo "=== Activando entorno ==="
source venv/bin/activate

echo "=== Instalando dependencias ==="
pip install \
  "Django>=5.2,<6.0" \
  "djangorestframework>=3.15,<4.0" \
  "mongoengine>=0.29,<1.0" \
  "pymongo>=4.9,<5.0" \
  "PyJWT>=2.9,<3.0" \
  "python-dotenv>=1.0,<2.0" \
  "django-cors-headers>=4.4,<5.0" \
  "drf-yasg>=1.21,<2.0" \
  "mongomock>=4.1,<5.0"

echo "=== Verificando instalaciones ==="
packages=(
  Django
  djangorestframework
  mongoengine
  pymongo
  PyJWT
  python-dotenv
  django-cors-headers
  drf-yasg
  mongomock
)

for pkg in "${packages[@]}"; do
  printf "%-20s => " "$pkg"
  pip show "$pkg" >/dev/null && echo "OK" || {
    echo "ERROR: $pkg no está instalado"
    exit 1
  }
done

echo "=== Todo instalado y verificado con éxito ==="
cd ./backend
python3 manage.py runserver