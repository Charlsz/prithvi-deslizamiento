"""Repite el protocolo ya fijado en otra carpeta, sin sobrescribir la evidencia."""
import argparse
from pathlib import Path
import shutil
import experimento_mejora as exp
import desarrollar_cabezas
import evaluar_extensiones
import visualizar_mejora
import verificar_mejora


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workdir',required=True)
    args=parser.parse_args();target=Path(args.workdir).resolve()
    original=exp.WORK
    if target==original.resolve():raise ValueError('Use otra carpeta para conservar la evaluación original.')
    if (target/'resultados').exists():raise ValueError('La carpeta ya contiene resultados; elija una carpeta nueva.')
    exp.WORK=target;exp.OUT=target/'resultados';exp.OUT.mkdir(parents=True,exist_ok=True)
    shutil.copy2(original/'resultados/protocolo.json',exp.OUT/'protocolo.json')
    p=exp.protocol()
    # Reutilizar exactamente los bytes descargados, si están disponibles.
    for entries in p['samples'].values():
        for entry in entries:
            for key in ['image','label']:
                src=original/'datos'/entry[key]
                if src.exists():
                    dst=target/'datos'/entry[key];dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    exp.develop(p)
    desarrollar_cabezas.main()
    exp.evaluate(p)
    evaluar_extensiones.main()
    visualizar_mejora.main()
    verificar_mejora.OUT=exp.OUT
    verificar_mejora.main()
    print('Reproducción completa:',target)


if __name__=='__main__':main()
