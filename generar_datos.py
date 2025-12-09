import csv
import random
import os

# Asegurar que la carpeta data existe
os.makedirs("data", exist_ok=True)

incidencias = [
    {"error": "ORA-12541: TNS:no listener", "causa": "El servicio de Oracle XE está detenido en el servidor local.", "solucion": "Reiniciar el servicio OracleServiceXE desde services.msc o ejecutar 'lsnrctl start' en la terminal."},
    {"error": "Connection refused at localhost:3000", "causa": "El microservicio de NestJS no se ha iniciado o se cayó.", "solucion": "Verificar logs de PM2 o ejecutar 'npm run start:dev' en el directorio del proyecto backend."},
    {"error": "MongoNetworkError: failed to connect to server", "causa": "La instancia de MongoDB no acepta conexiones entrantes o credenciales incorrectas.", "solucion": "Revisar cadena de conexión en .env y asegurar que mongod service esté activo."},
    {"error": "Heap out of memory in Node.js", "causa": "Fuga de memoria en el procesamiento de reporte masivo.", "solucion": "Aumentar memoria del proceso con --max-old-space-size=4096 o refactorizar el loop usando streams."},
    {"error": "COBOL File Status 35 on MAIN_DAT", "causa": "El archivo físico no existe en la ruta del mainframe especificada.", "solucion": "Verificar JCL de carga y asegurar que el dataset ha sido catalogado correctamente."},
    {"error": "ORA-00001: unique constraint violated", "causa": "Intento de insertar un ID de transacción duplicado.", "solucion": "Validar que el UUID se genere correctamente en el servicio antes del insert."},
    {"error": "NestJS: Cannot find module '@app/auth'", "causa": "Error en los alias del path en tsconfig.json tras actualización.", "solucion": "Revisar tsconfig.paths y borrar carpeta dist/ antes de recompilar."},
    {"error": "TimeoutError: ResourceRequest timed out", "causa": "El pool de conexiones a Oracle está saturado.", "solucion": "Aumentar UV_THREADPOOL_SIZE o incrementar connection limit en la config del TypeORM."}
]

filename = "data/base_conocimiento.csv"

with open(filename, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["ticket_id", "mensaje_error", "causa_raiz", "solucion_sugerida"])
    
    for i in range(1, 25):
        caso = random.choice(incidencias)
        writer.writerow([f"INC-{1000+i}", caso["error"], caso["causa"], caso["solucion"]])

print(f"Archivo '{filename}' generado exitosamente.")