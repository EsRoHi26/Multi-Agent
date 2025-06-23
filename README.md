# Multi-Agent

## Pasos para correrlo
1. Instalar las dependecias
   ```
   pip install -r requirements.txt
   ```
2. Configurar la variable del entorno con tu OpenAI key
   ```
   export OPENAI_API_KEY='tu-api-key'
   ```
3. Inicializar la base de datos vectorial (solo la primera vez)
   ```Python
   from database.vector_db import VectorDatabase
   db = VectorDatabase()
   db.initialize_db()
   ```
4. Correr el programa
   ```
   python main.py
   ```
