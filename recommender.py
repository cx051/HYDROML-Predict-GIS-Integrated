# ===============================
# BIS IS 10500:2012 Water Quality Standards
# ===============================

BIS_LIMITS = {
    "pH": {"acceptable": (6.5, 8.5), "permissible": (6.5, 8.5)},  # No relaxation
    "TDS": {"acceptable": 500, "permissible": 2000},
    "TH": {"acceptable": 200, "permissible": 600},
    "NO3": {"acceptable": 45, "permissible": 45},  # No relaxation
    "F": {"acceptable": 1.0, "permissible": 1.5},
    "Cl": {"acceptable": 250, "permissible": 1000},
    "SO4": {"acceptable": 200, "permissible": 400},
    "Ca": {"acceptable": 75, "permissible": 200},
    "Mg": {"acceptable": 30, "permissible": 100}
}

# Training Standards for WQI (Matches training script logic)
TRAINING_STANDARDS = {
    'pH': 8.5,
    'TDS': 500,
    'TH': 300,
    'Ca': 75,
    'Mg': 30,
    'Cl': 250,
    'SO4': 200,
    'NO3': 45,
    'F': 1.0
}

# ===============================
# Treatment Knowledge Base
# ===============================

TREATMENTS = {
    'TDS': ["Reverse Osmosis (RO)", "Nanofiltration"],
    'NO3': ["Ion Exchange", "Biological Denitrification"],
    'F': ["Activated Alumina", "Nalgonda Technique"],
    'TH': ["Lime-Soda Softening", "Ion Exchange"],
    'Cl': ["Reverse Osmosis"],
    'SO4': ["Reverse Osmosis"],
    'pH': ["pH Adjustment using Lime or Acid"],
    'Ca': ["Ion Exchange", "Lime Softening"],
    'Mg': ["Ion Exchange", "Lime Softening"]
}


# ===============================
# Recommendation Function
# ===============================

def generate_recommendations(sample_input, bis_results):
    """
    Generate tier-aware recommendations based on BIS compliance results.
    Acceptable: No treatment
    Permissible: Advisory treatment
    Violation: Mandatory treatment
    """
    recommendations = []

    for param, status in bis_results.items():
        if status == "acceptable":
            continue
            
        value = sample_input.get(param)

        # Specialized pH logic: Acidic vs Alkaline
        if param == "pH":
            if value < 6.5:
                recommendations.append({
                    "contaminant": "pH",
                    "value": value,
                    "status": status.capitalize(),
                    "severity": "Mandatory" if status == "violation" else "Advisory",
                    "treatment": [
                        "Lime Dosing (Calcium Hydroxide)",
                        "Soda Ash Addition (Sodium Carbonate)",
                        "Alkalinity Adjustment"
                    ],
                    "message": f"Water is acidic (pH {value}). Increase alkalinity to bring pH into safe range (6.5–8.5)."
                })
            elif value > 8.5:
                recommendations.append({
                    "contaminant": "pH",
                    "value": value,
                    "status": status.capitalize(),
                    "severity": "Mandatory" if status == "violation" else "Advisory",
                    "treatment": [
                        "Carbon Dioxide (CO₂) Dosing",
                        "Dilute Acid Addition (e.g., HCl)",
                        "Controlled Neutralization"
                    ],
                    "message": f"Water is alkaline (pH {value}). Controlled acid dosing required to reduce pH to safe range (6.5–8.5)."
                })
            continue

        treatment_list = TREATMENTS.get(param, [])
        
        if status == "violation":
            recommendations.append({
                "contaminant": param,
                "value": value,
                "status": "Violation",
                "severity": "Mandatory",
                "treatment": treatment_list,
                "message": f"CRITICAL: {param} level ({value}) exceeds permissible limits. Immediate corrective action required."
            })
        elif status == "permissible":
            recommendations.append({
                "contaminant": param,
                "value": value,
                "status": "Permissible",
                "severity": "Advisory",
                "treatment": treatment_list,
                "message": f"ADVISORY: {param} level ({value}) is within permissible range but exceeds acceptable limits. Monitoring or mild treatment recommended."
            })

    return recommendations
