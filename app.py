from flask import Flask, request, jsonify
import sympy as sp
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/solve-ode', methods=['POST'])
def solve_ode():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No se ha enviado información"}), 400

    equation_str = data.get("equation")
    if not equation_str:
        return jsonify({"error": "No se ha proporcionado la ecuación diferencial"}), 400

    # Si no se proporcionan condiciones, asignamos una condición por defecto: y(0)=1
    conditions = data.get("conditions", {})
    if not conditions:
        conditions = {"y0": "1"}

    try:
        x0_val = float(data.get("x0", 0))
        xEnd_val = float(data.get("xEnd", 5))
        h_val = float(data.get("h", 0.1))
    except Exception as e:
        return jsonify({"error": "Error al interpretar los parámetros numéricos", "detail": str(e)}), 400

    t = sp.symbols('t')
    y = sp.Function('y')(t)

    local_dict = {
        "diff": sp.diff,
        "y": y,
        "t": t
    }

    try:
        if "=" not in equation_str:
            raise ValueError("La ecuación debe contener el signo '=' para separar ambos lados.")
        lhs, rhs = equation_str.split("=")
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

    try:
        sol_expr = solution.rhs
        sol_func = sp.lambdify(t, sol_expr, modules=['numpy'])
        x_vals = np.arange(x0_val, xEnd_val + h_val, h_val)
        points = {"x": [float(x) for x in x_vals],
                  "y": [float(sol_func(x)) for x in x_vals]}
    except Exception as e:
        return jsonify({"error": "Error al generar puntos numéricos de la solución", "detail": str(e)}), 500

    solution_str = sp.pretty(solution)
    response = {
        "solution": solution_str,
        "method": used_method,
        "points": points
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
