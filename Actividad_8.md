# Universidad de la Salle
## Maestría en Inteligencia Artificial
### Fundamentos de Inteligencia Artificial — Actividad 8
**Realizado por:** Camilo Andrés Castañeda Galindo

---

## Caso de Estudio: Control de Calidad de Galletas mediante Visión Artificial

A partir del material de referencia de la unidad y de la actividad número 8 y especialmente del video ***Computer Vision con Python y OpenCV***, en donde de manera práctica, se implementa la visión por computador en python (OpenCV), que es una de las aplicaciones de inteligencia artificial y deep learning más usadas en la actualidad; se crea e implementa el siguiente caso de estudio:

**Problema:** Clasificar galletas en tiempo real como **aprobadas** o **quebradas** usando la cámara web y técnicas de procesamiento de imágenes con OpenCV.

**Métrica clave — Circularidad:**

$$\text{Circularidad} = \frac{4\pi \cdot A}{P^2}$$

| Circularidad | Clasificación |
|:---:|:---:|
| > 0.8 | Galleta aprobada (verde) |
| 0.5 – 0.8 | Galleta quebrada (rojo) |

---

## Pipeline del Sistema


| Etapa | Técnica | Propósito |
|:---|:---|:---|
| Preprocesamiento | Escala de grises + Gaussiano 11×11 | Reducir ruido y simplificar |
| Segmentación | Umbralización de Otsu | Separar galleta del fondo |
| Morfología | Cierre (3 iteración) + Apertura (1 iteración) | Rellenar huecos, eliminar ruido |
| Detección | `findContours` externos | Localizar silueta de cada galleta |
| Clasificación | Circularidad (4πA/P²) | Determinar si está entera o quebrada |

---

## Código Fuente - Python y OpenCV

```python
import cv2
import numpy as np

# Inicio de captura de video
capture = cv2.VideoCapture(0)

while True:
    ret, image = capture.read()

    # Preprocesamiento del video
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 3)

    # Segmentación por umbral
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Cerrar el borde dentado y rellenar hoyos de la galleta
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)

        # Filtrar: área grande + forma circular
        if area > 5000 and circularity > 0.5:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(image, center, 1, (255, 0, 0), 3)

            if circularity > 0.8:
                label = "Galleta aprobada"
                color = (0, 255, 0)
            else:
                label = "Galleta quebrada"
                color = (0, 0, 255)

            cv2.circle(image, center, radius, color, 3)
            cv2.putText(image, f"{label} ({circularity:.2f})", (center[0]-80, center[1]-radius-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Título
    cv2.putText(image, "Control de calidad de galletas", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Nombre
    h = image.shape[0]
    cv2.putText(image, "Creado por: Camilo A. Castaneda G.", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    cv2.imshow('video', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

---

## Resultados

### Galleta Aprobada — Circularidad = 0.89

![Galleta aprobada](galleta_aprobada.png)

La galleta presenta forma casi circular, indicando que está completa. El sistema la marca con borde **verde**.

### Galleta Quebrada — Circularidad = 0.71

![Galleta quebrada](galleta_quebrada.png)

La galleta tiene un fragmento faltante en el borde, reduciendo su circularidad. El sistema la marca con borde **rojo**.

---

## Conclusiones
- La visión por computador es una de las técnicas de inteligencia artificial, que más ampliamente están implementadas en el lenguaje Python, en este caso con la librería OpenCV, lo que hace que su implementación sea relativamente sencilla para proyectos básicos.
- La visión por computador permite hacer uso en tiempo real de aplicaciones de inteligencia artificial, que facilitan el apoyo en tareas que requieren atención y validación rápida, como por ejemplo en el control de calidad de productos o en la vigilancia de procesos industriales.
- A pesar de que la librería de OpenCV contiene grandes funcionalidades pre-programadas, no es suficiente para dar soluciones óptimas a casos de visión por computador, sin que en el procesos del desarrollo de software se ajusten los parámetros adecuados para evitar los falsos positivos y los falsos negativos.

## Referencias
Datademia. (2021, noviembre 28). ¿Qué es machine learning? [Video]. YouTube. https://www.youtube.com/watch?v=EDfI9K9egFc

Google Cloud Tech. (2018, abril 19). How computer vision works [Video]. YouTube. https://www.youtube.com/watch?v=OcycT1Jwsns

Heaton, J. (2022, mayo 25). Applications of deep neural networks for Keras [Video]. YouTube. https://www.youtube.com/watch?v=y9cQOMGiGek

IAAR. (s. f.). Pasos para construir un modelo de machine learning. En Introducción al Machine Learning. Recuperado el 12 de abril de 2026, de https://iaarbook.github.io/ML/#pasos-para-construir-un-modelo-de-machine-learning

Iacono, M. (2021, marzo 22). Computer vision con Python y OpenCV [Video]. YouTube. https://www.youtube.com/watch?v=ua25vHwca0E