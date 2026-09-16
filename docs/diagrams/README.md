# Diagramas

Esta carpeta contiene diagramas del proyecto en formato Mermaid.

## Esquema de datos

El esquema funcional para EDA, entrenamiento, monitoreo y predicciones esta en:

```text
docs/diagrams/data_schema.mmd
```

Para verlo rapido, copia el contenido en:

```text
https://mermaid.live
```

Para renderizarlo localmente con Mermaid CLI:

```bash
npm install -g @mermaid-js/mermaid-cli
mmdc -i docs/diagrams/data_schema.mmd -o docs/diagrams/data_schema.svg
mmdc -i docs/diagrams/data_schema.mmd -o docs/diagrams/data_schema.png
```
