from flask import Flask, request, jsonify
import sympy as sp
import numpy as np
from flask_cors import CORS  # Importa flask-cors

app = Flask(__name__)
CORS(app)  # Habilita CORS para todas las rutas

@app.route('/solve-ode', methods=['POST'])
def solve_ode():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No se ha enviado información"}), 400

    equation_str = data.get("equation")
    if not equation_str:
        return jsonify({"error": "No se ha proporcionado la ecuación diferencial"}), 400

    conditions = data.get("conditions", {})  # Ej: {"y0": "1", "yprime0": "0"}

    # Obtener parámetros numéricos para la generación de puntos
    try:
        x0_val = float(data.get("x0", 0))
        xEnd_val = float(data.get("xEnd", 5))
        h_val = float(data.get("h", 0.1))
    except Exception as e:
        return jsonify({"error": "Error al interpretar los parámetros numéricos", "detail": str(e)}), 400

    # Definir la variable independiente y la función dependiente
    t = sp.symbols('t')
    y = sp.Function('y')(t)

    # Diccionario de locales para sympify
    local_dict = {
        "diff": sp.diff,
        "y": y,
        "t": t
    }

    try:
        # Verificar que la ecuación incluya "="
        if "=" not in equation_str:
            raise ValueError("La ecuación debe contener el signo '=' para separar ambos lados.")
        lhs, rhs = equation_str.split("=")
        # Reemplazar notaciones de derivada
        lhs_processed = lhs.replace("y''", "diff(y, t, t)").replace("y'", "diff(y, t)")
        lhs_expr = sp.sympify(lhs_processed, locals=local_dict)
        rhs_expr = sp.sympify(rhs, locals=local_dict)
        eq = sp.Eq(lhs_expr, rhs_expr)
    except Exception as e:
        return jsonify({"error": "Error al interpretar la ecuación", "detail": str(e)}), 400

    ics = {}
    if "y0" in conditions:
        try:
            ic_y0 = sp.sympify(conditions["y0"])
            ics[y.subs(t, 0)] = ic_y0
        except Exception as e:
            return jsonify({"error": "Error al interpretar y(0)", "detail": str(e)}), 400

    if "yprime0" in conditions:
        try:
            ic_yprime0 = sp.sympify(conditions["yprime0"])
            ics[sp.diff(y, t).subs(t, 0)] = ic_yprime0
        except Exception as e:
            return jsonify({"error": "Error al interpretar y'(0)", "detail": str(e)}), 400

    try:
        if ics:
            solution = sp.dsolve(eq, y, ics=ics)
            used_method = "dsolve con condiciones iniciales (método automático)"
        else:
            solution = sp.dsolve(eq, y)
            used_method = "dsolve sin condiciones iniciales (método automático)"
    except Exception as e:
        return jsonify({"error": "Error al resolver la ecuación", "detail": str(e)}), 500

    # Extraer la expresión de la solución y generar puntos numéricos
    try:
        # Se asume que la solución es de la forma y(t) = <expresión>
        sol_expr = solution.rhs
        sol_func = sp.lambdify(t, sol_expr, modules=['numpy'])
        x_vals = np.arange(x0_val, xEnd_val + h_val, h_val)
        points = [{"x": float(x), "y": float(sol_func(x))} for x in x_vals]
    except Exception as e:
        return jsonify({"error": "Error al generar puntos numéricos de la solución", "detail": str(e)}), 500

    solution_str = sp.pretty(solution)
    response = {
        "solution": solution_str,
        "method": used_method,
        "points": points  # Puntos numéricos de la solución exacta
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
