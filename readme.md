# Gradient Descent from Scratch

> Implementación del descenso de gradiente **sin librerías de ML** — sólo matemáticas y NumPy.

---

## Motivación

La regresión lineal es probablemente el algoritmo de Machine Learning más conocido. Todo el mundo lo ha usado. La mayoría de la gente llama a `sklearn.fit()` y sigue adelante.

Y está bien, para producción o a la práctica, eso es exactamente lo que hay que hacer.

Pero hay algo que me llamaba la atención: **¿cuánta gente que usa regresión lineal ha derivado alguna vez la función de coste?** ¿Cuántos saben de dónde sale la regla de actualización de los parámetros θ, o por qué se usa `x²` en vez de `|x|`?

Este proyecto nació de esa curiosidad. Quería entender el algoritmo de verdad, desde las matemáticas, sin que ninguna librería me ocultara lo que pasa por dentro. Así que lo implementé desde cero, derivando las fórmulas originales a mano y traduciéndolas directamente a código.

Si tú también tienes esa curiosidad, este notebook es para ti.

---

## ¿Qué hay aquí?

Un notebook de Jupyter que cubre, paso a paso:

- **El gradiente** ∇f y por qué apunta en la dirección de mayor pendiente
- **La función de coste** J(θ) — demostración de por qué se usa el error cuadrático (y no el valor absoluto)
- **Derivación completa** de la regla de actualización, desde J(θ) hasta la fórmula final
- **Los tres tipos de descenso de gradiente**: Batch, Estocástico (SGD) y Mini-Batch
- **Visualizaciones**: superficie de coste en 3D, trayectoria del gradiente, curvas de convergencia, efecto del learning rate
- **Extensión a múltiples features** sin cambiar ni una línea del algoritmo

Todo sobre un dataset sintético de precios de casas, para mantenerlo sencillo y enfocado en el algoritmo.

---

## Fórmulas implementadas

Regla de actualización:

$$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)$$

Función de coste:

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2$$

Gradiente (derivado explícitamente en el notebook):

$$\frac{\partial J}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right) \cdot x_j^{(i)}$$

---

## Stack

```
Python 3.x
NumPy        ← toda la matemática
Matplotlib   ← visualizaciones
```

Nada de scikit-learn. Nada de TensorFlow. Nada de PyTorch. Si ves un `.fit()` que no hayamos escrito nosotros, algo ha ido mal.

---

## Uso

```bash
git clone https://github.com/tuusuario/gradient-descent-from-scratch
cd gradient-descent-from-scratch
pip install numpy matplotlib jupyter
jupyter notebook descenso_gradiente.ipynb
```

---

## Estructura

```
.
├── descenso_gradiente.ipynb   # El notebook principal
└── README.md
```

---

## Notas

Este proyecto está basado en las notas del curso de Machine Learning de Stanford (Andrew Ng). Si quieres profundizar más, es el mejor sitio por donde empezar.

El objetivo aquí no es velocidad ni eficiencia — es claridad. Cada función está escrita para que se entienda qué hace y por qué, no para que sea rápida.

---

*Para todos los que alguna vez han llamado a `.fit()` y se han preguntado qué pasa exactamente ahí dentro.*
