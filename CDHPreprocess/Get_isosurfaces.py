# generate the Molecular isosurface and Crystal isosurfaces.

import os
import glob
from chmpy.util.mesh import save_mesh
from chmpy import Crystal
from multiprocessing import Pool

def process_file(args):
    file, type, isovalue, separation, save_path = args
    file_id = file.split('/')[-1].split('.')[0]
    try:
        c = Crystal.load(file)
        if type == 'hirshfeld':
            surfaces = c.hirshfeld_surfaces()
        elif type == 'promolecule_density':
            surfaces = c.promolecule_density_isosurfaces()
        save_mesh(surfaces[0], f'{save_path}/{file_id}.ply')
    except Exception as e:
        print(f'error processing {file_id}: {str(e)}')

def main(args):
    data_dir = args['source_path']
    data_files = sorted(glob.glob(os.path.join(data_dir, '*.cif')))
    pool_args = [(file, args['type'], args['isovalue'], args['separation'],args['save_path']) for file in data_files]
    with Pool(args['num_processes']) as pool:
        pool.map(process_file, pool_args)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser('Dataset analysis')
    parser.add_argument('-sp', '--source-path', type=str,
                        default='./CIF')
    parser.add_argument('-np', '--num-processes', type=int, default=8)
    parser.add_argument('-t','--type', type=str, default='hirshfeld')

    args = parser.parse_args().__dict__
    
    save_path = f"./surfaces_data/{args['type']}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    args['save_path'] = save_path
    
    main(args=args)