import configparser
config = configparser.ConfigParser()

config['input'] = {'map_source':'websky', 'ksz_type':'ri', 'map_source':'maps/'+map_source+'/'}
config['output'] = {'output_data_path':'data/' }

config['experiments'] = {'Planck_SMICA':[45,5], 'CMB_S3':[7,1.4], 'CMB_S4':[1, 3]}
config['cmb_config'] = {'ellmin':30, 'ellmax':[3000, 4000, 4500]}
config['reconstruction_config'] = {}
