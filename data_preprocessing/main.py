import os
from argparse import ArgumentParser
from data import format_data

if __name__=='__main__':
    # ToDO: Agregar la opción de utilizar los datos de la carpeta de validación.
    parser = ArgumentParser()
    parser.add_argument('--output_directory', type=str, default='ner_data', help='Output directory to store the data generated')
    parser.add_argument('--seed', type=int, default=123, help='Seed used when shuffling examples')
    

    args = parser.parse_args()
    output_directory = args.output_directory
    seed = args.seed
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    print(f'Creating IOB2 format for the SocialDisNER task.')

    format_data(output_directory, seed)

    
