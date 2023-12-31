import numpy as np


def pgrid(r, order):
    match order:
        case 1:
            return r*np.array([
                    [0.00000000e+00, 0.00000000e+00],
                    [-8.09016994e-01, -5.87785252e-01],
                    [3.09016994e-01, -9.51056516e-01],
                    [1.00000000e+00, 0.00000000e+00],
                    [3.09016994e-01, 9.51056516e-01],
                    [-8.09016994e-01, 5.87785252e-01],
                    [-5.00000000e-01, -1.53884177e+00],
                    [1.30901699e+00, -9.51056516e-01],
                    [1.30901699e+00, 9.51056516e-01],
                    [-5.00000000e-01, 1.53884177e+00],
                    [-1.61803399e+00, 0.00000000e+00]
                    ])

        case 2:
            return r*np.array([
                [0.00000000e+00, 0.00000000e+00],
                [-8.09016994e-01, -5.87785252e-01],
                [3.09016994e-01, -9.51056516e-01],
                [1.00000000e+00, 0.00000000e+00],
                [3.09016994e-01, 9.51056516e-01],
                [-8.09016994e-01, 5.87785252e-01],
                [-1.30901699e+00, -9.51056516e-01],
                [-5.00000000e-01, -1.53884177e+00],
                [5.00000000e-01, -1.53884177e+00],
                [1.30901699e+00, -9.51056516e-01],
                [1.61803399e+00, 0.00000000e+00],
                [1.30901699e+00, 9.51056516e-01],
                [5.00000000e-01, 1.53884177e+00],
                [-5.00000000e-01, 1.53884177e+00],
                [-1.30901699e+00, 9.51056516e-01],
                [-1.61803399e+00, 1.11022302e-16],
                [-2.30901699e+00, -9.51056516e-01],
                [-1.61803399e+00, -1.90211303e+00],
                [1.90983006e-01, -2.48989828e+00],
                [1.30901699e+00, -2.12662702e+00],
                [2.42705098e+00, -5.87785252e-01],
                [2.42705098e+00, 5.87785252e-01],
                [1.30901699e+00, 2.12662702e+00],
                [1.90983006e-01, 2.48989828e+00],
                [-1.61803399e+00, 1.90211303e+00],
                [-2.30901699e+00, 9.51056516e-01],
                [-8.09016994e-01, -2.48989828e+00],
                [2.11803399e+00, -1.53884177e+00],
                [2.11803399e+00, 1.53884177e+00],
                [-8.09016994e-01, 2.48989828e+00],
                [-2.61803399e+00, 0.00000000e+00]])

        case 3:
            return r*np.array([
                [0.00000000e+00 ,    0.00000000e+00],
                [-3.09016994e-01,    -9.51056516e-01],
                [8.09016994e-01 ,   -5.87785252e-01],
                [8.09016994e-01 ,    5.87785252e-01],
                [-3.09016994e-01,     9.51056516e-01],
                [-1.00000000e+00,     1.37231113e-16],
                [-1.30901699e+00,    -9.51056516e-01],
                [5.00000000e-01 ,   -1.53884177e+00],
                [1.61803399e+00 ,   -2.22044605e-16],
                [5.00000000e-01 ,    1.53884177e+00],
                [-1.30901699e+00,     9.51056516e-01],
                [-1.80901699e+00,    -5.87785252e-01],
                [-1.11803399e+00,    -1.53884177e+00],
                [-1.11022302e-16,    -1.90211303e+00],
                [1.11803399e+00 ,   -1.53884177e+00],
                [1.80901699e+00 ,   -5.87785252e-01],
                [1.80901699e+00 ,    5.87785252e-01],
                [1.11803399e+00 ,    1.53884177e+00],
                [-1.79637859e-16,     1.90211303e+00],
                [-1.11803399e+00,     1.53884177e+00],
                [-1.80901699e+00,     5.87785252e-01],
                [-2.11803399e+00,    -1.53884177e+00],
                [-8.09016994e-01,    -2.48989828e+00],
                [8.09016994e-01 ,   -2.48989828e+00],
                [2.11803399e+00 ,   -1.53884177e+00],
                [2.61803399e+00 ,   -1.11022302e-16],
                [2.11803399e+00 ,    1.53884177e+00],
                [8.09016994e-01 ,    2.48989828e+00],
                [-8.09016994e-01,     2.48989828e+00],
                [-2.11803399e+00,     1.53884177e+00],
                [-2.61803399e+00,     0.00000000e+00],
                [-2.92705098e+00,    -9.51056516e-01],
                [-1.80901699e+00,    -2.48989828e+00],
                [-6.66133815e-16,    -3.07768354e+00],
                [1.80901699e+00 ,   -2.48989828e+00],
                [2.92705098e+00 ,   -9.51056516e-01],
                [2.92705098e+00 ,    9.51056516e-01],
                [1.80901699e+00 ,    2.48989828e+00],
                [-1.11022302e-16,     3.07768354e+00],
                [-1.80901699e+00,     2.48989828e+00],
                [-2.92705098e+00,     9.51056516e-01],
                [-3.23606798e+00,    -1.11022302e-16],
                [-1.00000000e+00,    -3.07768354e+00],
                [2.61803399e+00 ,   -1.90211303e+00],
                [2.61803399e+00 ,    1.90211303e+00],
                [-1.00000000e+00,     3.07768354e+00],
                [-2.92705098e+00,    -2.12662702e+00],
                [1.11803399e+00 ,   -3.44095480e+00],
                [3.61803399e+00 ,   -5.55111512e-17],
                [1.11803399e+00 ,    3.44095480e+00],
                [-2.92705098e+00,     2.12662702e+00],
                [-3.92705098e+00,    -9.51056516e-01],
                [-3.73606798e+00,    -1.53884177e+00],
                [-2.61803399e+00,    -3.07768354e+00],
                [-2.11803399e+00,    -3.44095480e+00],
                [-3.09016994e-01,    -4.02874005e+00],
                [3.09016994e-01 ,   -4.02874005e+00],
                [2.11803399e+00 ,   -3.44095480e+00],
                [2.61803399e+00 ,   -3.07768354e+00],
                [3.73606798e+00 ,   -1.53884177e+00],
                [3.92705098e+00 ,   -9.51056516e-01],
                [3.92705098e+00 ,    9.51056516e-01],
                [3.73606798e+00 ,    1.53884177e+00],
                [2.61803399e+00 ,    3.07768354e+00],
                [2.11803399e+00 ,    3.44095480e+00],
                [3.09016994e-01 ,    4.02874005e+00],
                [-3.09016994e-01,     4.02874005e+00],
                [-2.11803399e+00,     3.44095480e+00],
                [-2.61803399e+00,     3.07768354e+00],
                [-3.73606798e+00,     1.53884177e+00],
                [-3.92705098e+00,     9.51056516e-01],
                [-1.30901699e+00,    -4.02874005e+00],
                [3.42705098e+00 ,   -2.48989828e+00],
                [3.42705098e+00 ,    2.48989828e+00],
                [-1.30901699e+00,     4.02874005e+00],
                [-4.23606798e+00,     0.00000000e+00],
                [-4.73606798e+00,    -1.53884177e+00],
                [-2.92705098e+00,    -4.02874005e+00],
                [-6.10622664e-16,    -4.97979657e+00],
                [2.92705098e+00 ,   -4.02874005e+00],
                [4.73606798e+00 ,   -1.53884177e+00],
                [4.73606798e+00 ,    1.53884177e+00],
                [2.92705098e+00 ,    4.02874005e+00],
                [5.55111512e-16 ,    4.97979657e+00],
                [-2.92705098e+00,     4.02874005e+00],
                [-4.73606798e+00,     1.53884177e+00]
                ])

        case 4:
            return r*np.array([
                [0.00000000e+00,     0.00000000e+00],
                [-3.09016994e-01,    -9.51056516e-01],
                [8.09016994e-01 ,   -5.87785252e-01],
                [8.09016994e-01 ,    5.87785252e-01],
                [-3.09016994e-01,     9.51056516e-01],
                [-1.00000000e+00,     2.22044605e-16],
                [-1.30901699e+00,    -9.51056516e-01],
                [-5.00000000e-01,    -1.53884177e+00],
                [5.00000000e-01 ,   -1.53884177e+00],
                [1.30901699e+00 ,   -9.51056516e-01],
                [1.61803399e+00 ,   -3.59275718e-16],
                [1.30901699e+00 ,    9.51056516e-01],
                [5.00000000e-01 ,    1.53884177e+00],
                [-5.00000000e-01,     1.53884177e+00],
                [-1.30901699e+00,     9.51056516e-01],
                [-1.61803399e+00,     1.82140578e-16],
                [-2.42705098e+00,    -5.87785252e-01],
                [-1.30901699e+00,    -2.12662702e+00],
                [-1.90983006e-01,    -2.48989828e+00],
                [1.61803399e+00 ,   -1.90211303e+00],
                [2.30901699e+00 ,   -9.51056516e-01],
                [2.30901699e+00 ,    9.51056516e-01],
                [1.61803399e+00 ,    1.90211303e+00],
                [-1.90983006e-01,     2.48989828e+00],
                [-1.30901699e+00,     2.12662702e+00],
                [-2.42705098e+00,     5.87785252e-01],
                [-2.11803399e+00,    -1.53884177e+00],
                [8.09016994e-01 ,   -2.48989828e+00],
                [2.61803399e+00 ,    0.00000000e+00],
                [8.09016994e-01 ,    2.48989828e+00],
                [-2.11803399e+00,     1.53884177e+00],
                [-2.92705098e+00,    -9.51056516e-01],
                [-1.80901699e+00,    -2.48989828e+00],
                [-1.11022302e-16,    -3.07768354e+00],
                [1.80901699e+00 ,   -2.48989828e+00],
                [2.92705098e+00 ,   -9.51056516e-01],
                [2.92705098e+00 ,    9.51056516e-01],
                [1.80901699e+00 ,    2.48989828e+00],
                [-3.33066907e-16,     3.07768354e+00],
                [-1.80901699e+00,     2.48989828e+00],
                [-2.92705098e+00,     9.51056516e-01],
                [-3.23606798e+00,    -1.11022302e-16],
                [-1.00000000e+00,    -3.07768354e+00],
                [2.61803399e+00 ,   -1.90211303e+00],
                [2.61803399e+00 ,    1.90211303e+00],
                [-1.00000000e+00,     3.07768354e+00],
                [-3.11803399e+00,    -1.53884177e+00],
                [-2.42705098e+00,    -2.48989828e+00],
                [5.00000000e-01 ,   -3.44095480e+00],
                [1.61803399e+00 ,   -3.07768354e+00],
                [3.42705098e+00 ,   -5.87785252e-01],
                [3.42705098e+00 ,    5.87785252e-01],
                [1.61803399e+00 ,    3.07768354e+00],
                [5.00000000e-01 ,    3.44095480e+00],
                [-2.42705098e+00,     2.48989828e+00],
                [-3.11803399e+00,     1.53884177e+00],
                [-3.92705098e+00,    -9.51056516e-01],
                [-2.11803399e+00,    -3.44095480e+00],
                [-3.09016994e-01,    -4.02874005e+00],
                [2.61803399e+00 ,   -3.07768354e+00],
                [3.73606798e+00 ,   -1.53884177e+00],
                [3.73606798e+00 ,    1.53884177e+00],
                [2.61803399e+00 ,    3.07768354e+00],
                [-3.09016994e-01,     4.02874005e+00],
                [-2.11803399e+00,     3.44095480e+00],
                [-3.92705098e+00,     9.51056516e-01],
                [-3.42705098e+00,    -2.48989828e+00],
                [-1.30901699e+00,    -4.02874005e+00],
                [1.30901699e+00 ,   -4.02874005e+00],
                [3.42705098e+00 ,   -2.48989828e+00],
                [4.23606798e+00 ,   -3.59275718e-16],
                [3.42705098e+00 ,    2.48989828e+00],
                [1.30901699e+00 ,    4.02874005e+00],
                [-1.30901699e+00,     4.02874005e+00],
                [-3.42705098e+00,     2.48989828e+00],
                [-4.23606798e+00,     0.00000000e+00],
                [-4.23606798e+00,    -1.90211303e+00],
                [-3.11803399e+00,    -3.44095480e+00],
                [5.00000000e-01 ,   -4.61652531e+00],
                [2.30901699e+00 ,   -4.02874005e+00],
                [4.54508497e+00 ,   -9.51056516e-01],
                [4.54508497e+00 ,    9.51056516e-01],
                [2.30901699e+00 ,    4.02874005e+00],
                [5.00000000e-01 ,    4.61652531e+00],
                [-3.11803399e+00,     3.44095480e+00],
                [-4.23606798e+00,     1.90211303e+00],
                [-4.73606798e+00,    -1.53884177e+00],
                [-2.92705098e+00,    -4.02874005e+00],
                [-1.22124533e-15,    -4.97979657e+00],
                [2.92705098e+00 ,   -4.02874005e+00],
                [4.73606798e+00 ,   -1.53884177e+00],
                [4.73606798e+00 ,    1.53884177e+00],
                [2.92705098e+00 ,    4.02874005e+00],
                [-2.22044605e-16,     4.97979657e+00],
                [-2.92705098e+00,     4.02874005e+00],
                [-4.73606798e+00,     1.53884177e+00],
                [-5.04508497e+00,    -5.87785252e-01],
                [-2.11803399e+00,    -4.61652531e+00],
                [-1.00000000e+00,    -4.97979657e+00],
                [3.73606798e+00 ,   -3.44095480e+00],
                [4.42705098e+00 ,   -2.48989828e+00],
                [4.42705098e+00 ,    2.48989828e+00],
                [3.73606798e+00 ,    3.44095480e+00],
                [-1.00000000e+00,     4.97979657e+00],
                [-2.11803399e+00,     4.61652531e+00],
                [-5.04508497e+00,     5.87785252e-01],
                [-5.23606798e+00,    -8.98189294e-17],
                [-4.23606798e+00,    -3.07768354e+00],
                [-1.61803399e+00,    -4.97979657e+00],
                [1.61803399e+00 ,   -4.97979657e+00],
                [4.23606798e+00 ,   -3.07768354e+00],
                [5.23606798e+00 ,   -1.11022302e-16],
                [4.23606798e+00 ,    3.07768354e+00],
                [1.61803399e+00 ,    4.97979657e+00],
                [-1.61803399e+00,     4.97979657e+00],
                [-4.23606798e+00,     3.07768354e+00],
                [-5.04508497e+00,    -2.48989828e+00],
                [-3.92705098e+00,    -4.02874005e+00],
                [8.09016994e-01 ,   -5.56758182e+00],
                [2.61803399e+00 ,   -4.97979657e+00],
                [5.54508497e+00 ,   -9.51056516e-01],
                [5.54508497e+00 ,    9.51056516e-01],
                [2.61803399e+00 ,    4.97979657e+00],
                [8.09016994e-01 ,    5.56758182e+00],
                [-3.92705098e+00,     4.02874005e+00],
                [-5.04508497e+00,     2.48989828e+00],
                [-4.73606798e+00,    -3.44095480e+00],
                [1.80901699e+00 ,   -5.56758182e+00],
                [5.85410197e+00 ,   -2.22044605e-16],
                [1.80901699e+00 ,    5.56758182e+00],
                [-4.73606798e+00,     3.44095480e+00],
                [-5.73606798e+00,    -1.53884177e+00],
                [-3.23606798e+00,    -4.97979657e+00],
                [-3.09016994e-01,    -5.93085309e+00],
                [3.73606798e+00 ,   -4.61652531e+00],
                [5.54508497e+00 ,   -2.12662702e+00],
                [5.54508497e+00 ,    2.12662702e+00],
                [3.73606798e+00 ,    4.61652531e+00],
                [-3.09016994e-01,     5.93085309e+00],
                [-3.23606798e+00,     4.97979657e+00],
                [-5.73606798e+00,     1.53884177e+00],
                [-6.04508497e+00,    -5.87785252e-01],
                [-2.42705098e+00,    -5.56758182e+00],
                [-1.30901699e+00,    -5.93085309e+00],
                [4.54508497e+00 ,   -4.02874005e+00],
                [5.23606798e+00 ,   -3.07768354e+00],
                [5.23606798e+00 ,    3.07768354e+00],
                [4.54508497e+00 ,    4.02874005e+00],
                [-1.30901699e+00,     5.93085309e+00],
                [-2.42705098e+00,     5.56758182e+00],
                [-6.04508497e+00,     5.87785252e-01],
                [-6.35410197e+00,    -1.53884177e+00],
                [-6.04508497e+00,    -2.48989828e+00],
                [-4.23606798e+00,    -4.97979657e+00],
                [-3.42705098e+00,    -5.56758182e+00],
                [-5.00000000e-01,    -6.51863834e+00],
                [5.00000000e-01 ,   -6.51863834e+00],
                [3.42705098e+00 ,   -5.56758182e+00],
                [4.23606798e+00 ,   -4.97979657e+00],
                [6.04508497e+00 ,   -2.48989828e+00],
                [6.35410197e+00 ,   -1.53884177e+00],
                [6.35410197e+00 ,    1.53884177e+00],
                [6.04508497e+00 ,    2.48989828e+00],
                [4.23606798e+00 ,    4.97979657e+00],
                [3.42705098e+00 ,    5.56758182e+00],
                [5.00000000e-01 ,    6.51863834e+00],
                [-5.00000000e-01,     6.51863834e+00],
                [-3.42705098e+00,     5.56758182e+00],
                [-4.23606798e+00,     4.97979657e+00],
                [-6.04508497e+00,     2.48989828e+00],
                [-6.35410197e+00,     1.53884177e+00],
                [-5.73606798e+00,    -3.44095480e+00],
                [-5.04508497e+00,    -4.39201132e+00],
                [1.50000000e+00 ,   -6.51863834e+00],
                [2.61803399e+00 ,   -6.15536707e+00],
                [6.66311896e+00 ,   -5.87785252e-01],
                [6.66311896e+00 ,    5.87785252e-01],
                [2.61803399e+00 ,    6.15536707e+00],
                [1.50000000e+00 ,    6.51863834e+00],
                [-5.04508497e+00,     4.39201132e+00],
                [-5.73606798e+00,     3.44095480e+00],
                [-2.11803399e+00,    -6.51863834e+00],
                [5.54508497e+00 ,   -4.02874005e+00],
                [5.54508497e+00 ,    4.02874005e+00],
                [-2.11803399e+00,     6.51863834e+00],
                [-6.85410197e+00,     0.00000000e+00],
                [-6.85410197e+00,    -1.90211303e+00],
                [-3.92705098e+00,    -5.93085309e+00],
                [-3.09016994e-01,    -7.10642359e+00],
                [4.42705098e+00 ,   -5.56758182e+00],
                [6.66311896e+00 ,   -2.48989828e+00],
                [6.66311896e+00 ,    2.48989828e+00],
                [4.42705098e+00 ,    5.56758182e+00],
                [-3.09016994e-01,     7.10642359e+00],
                [-3.92705098e+00,     5.93085309e+00],
                [-6.85410197e+00,     1.90211303e+00],
                [-7.16311896e+00,    -9.51056516e-01],
                [-3.11803399e+00,    -6.51863834e+00],
                [-1.30901699e+00,    -7.10642359e+00],
                [5.23606798e+00 ,   -4.97979657e+00],
                [6.35410197e+00 ,   -3.44095480e+00],
                [6.35410197e+00 ,    3.44095480e+00],
                [5.23606798e+00 ,    4.97979657e+00],
                [-1.30901699e+00,     7.10642359e+00],
                [-3.11803399e+00,     6.51863834e+00],
                [-7.16311896e+00,     9.51056516e-01],
                [-6.85410197e+00,    -3.07768354e+00],
                [-5.04508497e+00,    -5.56758182e+00],
                [8.09016994e-01 ,   -7.46969485e+00],
                [3.73606798e+00 ,   -6.51863834e+00],
                [7.35410197e+00 ,   -1.53884177e+00],
                [7.35410197e+00 ,    1.53884177e+00],
                [3.73606798e+00 ,    6.51863834e+00],
                [8.09016994e-01 ,    7.46969485e+00],
                [-5.04508497e+00,     5.56758182e+00],
                [-6.85410197e+00,     3.07768354e+00],
                [-7.66311896e+00,    -2.48989828e+00],
                [-4.73606798e+00,    -6.51863834e+00],
                [-8.88178420e-16,    -8.05748011e+00],
                [4.73606798e+00 ,   -6.51863834e+00],
                [7.66311896e+00 ,   -2.48989828e+00],
                [7.66311896e+00 ,    2.48989828e+00],
                [4.73606798e+00 ,    6.51863834e+00],
                [9.99200722e-16 ,    8.05748011e+00],
                [-4.73606798e+00,     6.51863834e+00],
                [-7.66311896e+00,     2.48989828e+00]
                ])
