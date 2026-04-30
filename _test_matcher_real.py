"""
Test directo del matcher con tus datos reales.
Ejecutar: python3 _test_matcher_real.py
"""
from core.sensor_map import resolve_sensor_for_point, sensor_label

# Sensor map del usuario
sensors = [
    {'plane': 1, 'plane_label': 'CRF Accel', 'side': 'R', 'angle_deg': 135.0,
     'direction': 'radial', 'sensor_type': 'accelerometer', 'unit_native': 'g peak',
     'alarm': 3.0, 'danger': 6.0,
     'csv_match_pattern': '*acell*acell*, *1*acell*acell*'},
    {'plane': 1, 'plane_label': 'CRF Vel', 'side': 'R', 'angle_deg': 135.0,
     'direction': 'radial', 'sensor_type': 'velocity', 'unit_native': 'in/s peak',
     'alarm': 1.5, 'danger': 2.0,
     'csv_match_pattern': '*acell*vel*, *1*acell*vel*'},
    {'plane': 2, 'plane_label': 'TRF Accel', 'side': 'R', 'angle_deg': 135.0,
     'direction': 'radial', 'sensor_type': 'accelerometer', 'unit_native': 'g peak',
     'alarm': 3.0, 'danger': 6.0,
     'csv_match_pattern': '*acell*acell*, *2*acell*acell*'},
    {'plane': 2, 'plane_label': 'TRF Vel', 'side': 'R', 'angle_deg': 135.0,
     'direction': 'radial', 'sensor_type': 'velocity', 'unit_native': 'in/s peak',
     'alarm': 1.5, 'danger': 2.0,
     'csv_match_pattern': '*acell*vel*, *2*acell*vel*'},
]

# Datos REALES de tus CSVs
tests = [
    ('1VT6805 (C) TRF',  'Vel Wf(64X/32revs).KPHGEN',  'in/s'),
    ('1VT6831 (C) CRF',  'Vel Wf(64X/32revs).KPHGEN',  'in/s'),
    ('CRF ACELL',        'Accl Wf(1000Hz)',            'g'),
    ('TRF ACELL',        'Accl Wf(1000Hz)',            'g'),
]
print('Test del matcher con TUS Points reales:')
for pt, var, unit in tests:
    s = resolve_sensor_for_point(sensors, pt, var, unit)
    if s:
        ptype = s.get('sensor_type', '?')
        plbl = s.get('plane_label', '?')
        a = s.get('alarm', 0)
        d = s.get('danger', 0)
        un = s.get('unit_native', '?')
        ok = "✓" if (
            ('VT' in pt and ptype == 'velocity')
            or ('ACELL' in pt and ptype == 'accelerometer')
        ) else "✗ FALLA"
        print(f'  {pt:24s} → {sensor_label(s):8s} {ptype:13s} A={a:.2f} D={d:.2f} {un}  {ok}')
    else:
        print(f'  {pt:24s} → SIN MATCH ✗')
