from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="fa",
    ext_modules=[
        CUDAExtension(
            "fa", ["fa.cpp", "flash_attention.cu"], extra_compile_args={"nvcc": ["-O2", "--ptxas-options=-v"]}
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
