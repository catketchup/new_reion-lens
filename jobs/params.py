runtime = '00:40:00'



experiments = {
    'Planck_SMICA': {
        'nlev_t': 45,
        'beam_arcmin': 5
    },
    'CMB_S3': {
        'nlev_t': 7,
        'beam_arcmin': 1.4
    },
    'CMB_S4': {
        'nlev_t': 1,
        'beam_arcmin': 3
    },
    'reference': {
        'nlev_t': 0,
        'beam_arcmin': 3
    }
}

moments = {'moments1':{'ellmin':30, 'ellmax':3000, 'delta_L':150},'moments2':{'ellmin':30, 'ellmax':4000, 'delta_L':200}}