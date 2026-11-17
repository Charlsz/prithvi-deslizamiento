"""Ejecuta el cuaderno completo con un kernel nuevo y conserva salidas reales."""
from pathlib import Path
import os
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parent

def main():
    path = ROOT / 'Grupo_2_Trabajo_Deslizamientos.ipynb'
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    # Kernel explícito del mismo intérprete, sin depender de un kernel global.
    kernel_dir = ROOT / 'tmp' / 'revision' / 'kernels' / 'prithvi-local'
    kernel_dir.mkdir(parents=True, exist_ok=True)
    import json
    (kernel_dir / 'kernel.json').write_text(json.dumps({
        'argv': [sys.executable, '-X', 'utf8', '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'Prithvi local', 'language': 'python'
    }), encoding='utf-8')
    manager = KernelSpecManager(kernel_dirs=[str(kernel_dir.parent)])
    client = NotebookClient(notebook, timeout=1800, kernel_name='prithvi-local',
                            resources={'metadata': {'path': str(ROOT)}})
    client.km = client.create_kernel_manager()
    client.km.kernel_spec_manager = manager
    print('Ejecutando todas las celdas con kernel nuevo...', flush=True)
    client.execute()
    temp = path.with_suffix('.executed.ipynb')
    nbformat.write(notebook, temp)
    temp.replace(path)
    print('Cuaderno ejecutado y guardado sin errores.', flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
