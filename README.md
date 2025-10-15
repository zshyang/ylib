# ylib

## Installation

```bash
git clone https://github.com/zshyang/ylib.git
cd ylib
pip install -e .
```

## 1. Joint Skeleton OBJ Writer

To generate a demo skeleton mesh, run:

```bash
python scripts/00_joint_drawer.py
```

After running the command, the output file will be saved at: `demo_output/skeleton.obj`.

You can open it in any 3D viewer (e.g., Blender or MeshLab) to inspect the generated skeleton.
