from flask import Flask, request, jsonify
import sympy as sp
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def format_solution(solution_str: str) -> str:
    """Convierte la solución de SymPy a notación matemática legible"""
    replacements = {
        "exp(t)": "e^t",
        "exp(-t)": "e^{-t}",
        "1.0*": "",
        "0.5*": "½",
        "sqrt": "√",
        "**": "^",
        "exp": "e^",
        "cos": "cos",
        "sin": "sen",
        "Eq(y(t), ": "y(t) = ",
        ")": " )"
    }
    
    formatted = solution_str
    for k, v in replacements.items():
        formatted = formatted.replace(k, v)
    
    if "/" in formatted:
        parts = formatted.split("/")
        if len(parts) == 2 and not any(c in parts[1] for c in "+-*"):
            formatted = f"{parts[0]}/{parts[1]}"
    
    return formatted.replace("(t)", "").strip()

def extract_f(eq, y_func, t):
    deriv = sp.diff(y_func, t)
    sol = sp.solve(eq, deriv)
    if not sol:
        return None
    y_sym = sp.symbols('y')
    return sp.simplify(sol[0]).subs(y_func, y_sym)

@app.route('/solve-ode', methods=['POST'])
def solve_ode():
    try:
        data = request.get_json()
        equation_str = data.get("equation", "")
        
        if not equation_str or "=" not in equation_str:
            return jsonify({"error": "Ecuación no válida"}), 400

        # Parámetros numéricos
        x0 = float(data.get("x0", 0))
        x_end = float(data.get("xEnd", 5))
        h = float(data.get("h", 0.1))
        y0 = float(data.get("conditions", {}).get("y0", 1))

        # Parsear ecuación
        t = sp.symbols('t')
        y = sp.Function('y')(t)
        lhs, rhs = equation_str.split("=", 1)
        
        lhs_parsed = sp.sympify(
            lhs.strip()
            .replace("y''", "diff(y, t, t)")
            .replace("y'", "diff(y, t)"),
            locals={'diff': sp.diff, 'y': y, 't': t}
        )
        rhs_parsed = sp.sympify(rhs.strip(), locals={'y': y, 't': t})
        eq = sp.Eq(lhs_parsed, rhs_parsed)

        # Extraer función f(t,y)
        f_expr = extract_f(eq, y, t)
        if not f_expr:
            return jsonify({"error": "No se pudo despejar y'"}), 400

        # Resolver ecuación exacta
        ics = {y.subs(t, x0): y0}
        sol = sp.dsolve(eq, y, ics=ics)
        sol_func = sp.lambdify(t, sol.rhs, 'numpy')
        x_vals = np.arange(x0, x_end + h, h)
        exact_vals = np.vectorize(sol_func)(x_vals)

        # Configurar función numérica
        y_sym = sp.symbols('y')
        f_func = sp.lambdify((t, y_sym), f_expr, 'numpy')

        # Inicializar métodos
        methods = {
            'euler': [y0],
            'improved': [y0],
            'rk': [y0]
        }

        # Calcular métodos numéricos
        for i in range(1, len(x_vals)):
            # Euler
            y_euler = methods['euler'][-1]
            methods['euler'].append(y_euler + h * f_func(x_vals[i-1], y_euler))
            
            # Euler Mejorado
            y_imp = methods['improved'][-1]
            k1 = f_func(x_vals[i-1], y_imp)
            k2 = f_func(x_vals[i-1] + h, y_imp + h * k1)
            methods['improved'].append(y_imp + h * (k1 + k2) / 2)
            
            # Runge-Kutta
            y_rk = methods['rk'][-1]
            k1_rk = f_func(x_vals[i-1], y_rk)
            k2_rk = f_func(x_vals[i-1] + h/2, y_rk + h/2 * k1_rk)
            k3_rk = f_func(x_vals[i-1] + h/2, y_rk + h/2 * k2_rk)
            k4_rk = f_func(x_vals[i-1] + h, y_rk + h * k3_rk)
            methods['rk'].append(y_rk + h * (k1_rk + 2*k2_rk + 2*k3_rk + k4_rk) / 6)

        # Calcular errores
        errors = {
            method: [
                abs((exact - approx)/exact * 100) if exact != 0 else 0 
                for exact, approx in zip(exact_vals, methods[method])
            ]
            for method in methods
        }

        return jsonify({
            "solution": format_solution(str(sol)),
            "exact": {"x": x_vals.tolist(), "y": exact_vals.tolist()},
            "methods": {method: {"x": x_vals.tolist(), "y": vals} for method, vals in methods.items()},
            "errors": {method: {"x": x_vals.tolist(), "y": errs} for method, errs in errors.items()}
        })

    except Exception as e:
        return jsonify({"error": f"Error interno: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)