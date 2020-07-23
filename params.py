runtime = '00:30:00'

# maps information
map_source = 'Colin'
ksz_type = 'lt'
decmax = 45
width_deg = 30
px_arcmin = 1
# cutouts number
cutouts = int(2*decmax/width_deg*(360/width_deg))

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
