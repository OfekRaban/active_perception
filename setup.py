from setuptools import setup, find_packages

setup(
    name="active_perception",
    version="0.1.0",
    description="Active Perception via Latent External Visual Memory for MLLMs",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.45.0",
        "Pillow>=9.0",
        "pyyaml>=6.0",
        "pandas>=1.5",
        "pyarrow>=11.0",
        "tqdm>=4.65",
    ],
    extras_require={
        "lora": ["peft>=0.7.0"],
        "dev": ["pytest>=7.0", "pytest-cov"],
    },
)
