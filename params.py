# nersc, interactive or not
interactive = 'True'
runtime = '00:05:00'

# add beam at first?
beam = False

# use Gaussian ksz or not
use_ksz_g = True

# maps information, 'Colin' or 'websky'
map_source = 'Colin'
ksz_type = 'lt'
decmax = 45
width_deg = 10
px_arcmin = 1
path = '/global/cscratch1/sd/hongbo/new_reion-lens/'
map_path = path + 'maps/' + map_source + '/'

# alms fits files
cmb_alms_file = map_path + 'alms/' + 'lensed_cmb_alms_7000.fits'
ksz_alms_file = map_path +  'alms/' + f'ksz_{ksz_type}_alms_7000.fits'
ksz_g_alms_files = map_path + 'alms/' + f'ksz_g_{ksz_type}_alms_7000.fits'
kap_alms_file = map_path + 'alms/' + 'kap_alms_7000.fits'

# ksz_cls
ksz_cls_file = map_path + 'cls/' + f'ksz_{ksz_type}_cls_7000.csv'


# cutouts number
cutouts = int(2 * decmax / width_deg * (360 / width_deg))

# output data path
data_path = 'output/data' + str(cutouts) + '/'

# experiments
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
    },
    'test': {
        'nlev_t': 30,
        'beam_arcmin': 7
    }
}

moments = {'moments1':{'ellmin':30, 'ellmax':3000, 'delta_L':20},'moments2':{'ellmin':30, 'ellmax':4000, 'delta_L':20}}

# moments = {'moments1':{'ellmin':30, 'ellmax':5000, 'delta_L':150}}
