# Radar Fútbol — versión final PWA

Aplicación de predicción prepartido para iPhone, Android y Mac Sonoma.

## Mercados incluidos

- Ganador: 1 / X / 2
- Doble Oportunidad: 1X / X2 / 12
- Ambos Marcan: Sí / No
- Tarjetas Amarillas: + / −
- Tarjetas Rojas: + / −
- Goles Totales: + / −

Toda la interfaz está en español. La forma usa:
- G = Ganado
- E = Empatado
- P = Perdido

La hora se muestra siempre en horario de Chile (`America/Santiago`) y formato 24 horas.

## Actualización

La pantalla principal fuerza una actualización al abrir y al pulsar **Actualizar datos ahora**.
El backend vuelve a consultar las fuentes. La frescura final depende de la frecuencia con la que el proveedor
actualiza cada endpoint.

API-FOOTBALL indica que su cobertura puede variar por liga, temporada y partido. Sus datos de fixtures/eventos
pueden actualizarse con alta frecuencia; posiciones, lesiones, predicciones y cuotas tienen cadencias propias.

## Fuentes

### Datos deportivos principales
API-FOOTBALL / API-SPORTS:
- fixtures
- standings
- team statistics
- injuries
- lineups
- odds
- predictions

Se necesita:
`API_SPORTS_KEY`

### Contexto reciente opcional
OpenAI Responses API con búsqueda web:
- bajas/sanciones recientes
- cambios de técnico
- estadísticas del árbitro
- VAR cuando existe designación pública

Se necesita:
`OPENAI_API_KEY`

La clave OpenAI sólo se usa en el servidor. Nunca se expone al navegador o la PWA.

## Mac Sonoma

1. Descomprime la carpeta.
2. Edita `.env` y agrega `API_SPORTS_KEY`.
3. Si usarás enriquecimiento OpenAI, agrega `OPENAI_API_KEY`.
4. Doble clic en `INICIAR_EN_MAC.command`.
5. La app abre en `http://127.0.0.1:8787`.

Si macOS bloquea el script la primera vez:
clic derecho → Abrir.

## iPhone / Android

La app es una PWA. Para instalarla en teléfonos debe estar publicada por HTTPS.

iPhone:
1. Abre la URL HTTPS en Safari.
2. Compartir.
3. **Añadir a pantalla de inicio**.

Android:
1. Abre la URL HTTPS en Chrome.
2. Menú.
3. **Instalar aplicación** o **Añadir a pantalla de inicio**.

## Publicación HTTPS

Se incluye `Dockerfile` y `render.yaml`.
Configura las variables de entorno en el servicio de hosting:
- API_SPORTS_KEY
- OPENAI_API_KEY (opcional)
- OPENAI_MODEL=gpt-5.6
- MAX_FIXTURES_ANALYZE=30

## Modelo

No utiliza prestigio histórico como variable.

Pondera información actual:
- forma actual
- tabla de posiciones
- puntos por partido
- ataque/defensa
- localía
- predicción del proveedor
- cuotas disponibles
- lesiones y alineaciones cuando existen
- árbitro/contexto reciente cuando OpenAI está configurado

El Top diario ordena por la probabilidad más alta encontrada entre los seis mercados solicitados.

## Nota

Las probabilidades son estimaciones estadísticas y no garantizan resultados.
