from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="black-hole-simulation",
    version="1.0.0",
    author="Black Hole Simulation Contributors",
    description="A real-time 3D black hole simulation with gravitational physics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/black-hole-simulation",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "black-hole-sim=main:main",
        ],
    },
    keywords="physics simulation black-hole pygame visualization",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/black-hole-simulation/issues",
        "Source": "https://github.com/yourusername/black-hole-simulation",
        "Documentation": "https://github.com/yourusername/black-hole-simulation#readme",
    },
)